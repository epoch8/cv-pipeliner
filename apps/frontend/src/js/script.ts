import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

import Webcam from './webcam';
import { Document } from './interfaces';
import { BOX_COLORS, Url } from './constants';
import { drawBox, drawLabel } from './helpers';

const webcamElement = document.querySelector('.video') as HTMLVideoElement;

const canvasLayers = document.querySelector('.canvas-layers') as HTMLDivElement;
const canvasImage = document.querySelector('.image-layer') as HTMLCanvasElement;
const canvasBoxes = document.querySelector('.boxes-layer') as HTMLCanvasElement;
const boxesContext = canvasBoxes.getContext('2d') as CanvasRenderingContext2D;

const inputFps = document.querySelector('.input-field--fps input') as HTMLInputElement;
const inputQuality = document.querySelector('.input-field--quality input') as HTMLInputElement;
const inputDelay = document.querySelector('.input-field--delay input') as HTMLInputElement;

const buttonFlipCamera = document.querySelector('.button--camera-flip') as HTMLButtonElement;
const buttonStart = document.querySelector('.button--start') as HTMLButtonElement;
const buttonFullScreen = document.querySelector('.button--fullscreen') as HTMLButtonElement;
const buttonDownload = document.querySelector('.button--download') as HTMLButtonElement;

const webcam = new Webcam(webcamElement, 'environment', canvasImage);

const colorCache: { [key: string]: string } = {};

let cancelTokenSource = null;

const MAX_FPS = 60;
const DEFAULT_FPS = 5;
const MAX_QUALITY = 1;
const DEFAULT_QUALITY = 0.92;
const DEFAULT_DELAY = 200;
const SECOND = 1000;

let fps = +localStorage.getItem('fps') || DEFAULT_FPS;
let quality = +localStorage.getItem('quality') || DEFAULT_QUALITY;
let delay = +localStorage.getItem('delay') || DEFAULT_DELAY;

let snapImageTimeoutId;
let snapBoxesTimeoutId;

let isStreaming = false;

let guid = null;

inputFps.value = `${fps}`;
inputQuality.value = `${quality}`;
inputDelay.value = `${delay}`;

const realtimeStart = async () => {
  guid = uuidv4();
  const formData = new FormData();
  formData.append('fps', `${fps}`);
  formData.append('detection_delay', `${delay}`);

  try {
    await axios.post(`${Url.REALTIME_START}/${guid}`, formData);
  } catch (err) {
    console.error('realtimeStart', err);
  }
};

const realtimeEnd = async () => {
  try {
    await axios.post(`${Url.REALTIME_END}/${guid}`);
    guid = null;
  } catch (err) {
    console.error('realtimeEnd', err);
  }
};

const snapImage = () => {
  if (!isStreaming) {
    return;
  }

  snapImageTimeoutId = setTimeout(() => {
    try {
      webcam.snap({ quality: Number(quality) });
      snapImage();
    } catch (err) {
      console.error(err);
      snapImage();
    }
  }, SECOND / (Number(fps) || DEFAULT_FPS));
};

const snapBoxes = () => {
  if (!isStreaming) {
    return;
  }

  snapBoxesTimeoutId = setTimeout(() => {
    canvasImage.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append('image', blob);
      cancelTokenSource = axios.CancelToken.source();

      try {
        const { data: { bboxes_data } } = await axios.post(`${Url.REALTIME_PREDICT}/${guid}`, formData, {
          cancelToken: cancelTokenSource.token,
        });

        boxesContext.clearRect(
          0,
          0,
          canvasBoxes.width,
          canvasBoxes.height,
        );

        bboxes_data.forEach((box, index) => {
          if (!colorCache[box.label]) {
            colorCache[box.label] = BOX_COLORS[index];
          }

          drawBox(boxesContext, box, colorCache[box.label]);
          drawBox(boxesContext, box, 'black', 1);
          drawLabel(boxesContext, box, box.label, colorCache[box.label]);
        });
        snapBoxes();
      } catch (err) {
        if (!axios.isCancel(err)) {
          console.error(err);
          snapBoxes();
        }
      }
    }, 'image/jpeg');
  }, SECOND / (Number(fps) || DEFAULT_FPS));
};

const clearTimeouts = () => {
  clearTimeout(snapImageTimeoutId);
  snapImageTimeoutId = null;

  clearTimeout(snapBoxesTimeoutId);
  snapBoxesTimeoutId = null;
};

const startStream = async () => {
  isStreaming = true;

  buttonStart.textContent = 'Pause';
  buttonFlipCamera.disabled = webcam.webcamCount === 1;
  buttonFullScreen.disabled = false;
  buttonDownload.disabled = false;

  await webcam.start();

  if (guid) {
    await realtimeEnd();
  }

  await realtimeStart();

  canvasBoxes.height = webcamElement.scrollHeight;
  canvasBoxes.width = webcamElement.scrollWidth;

  snapImage();
  snapBoxes();
};

const stopStream = async () => {
  clearTimeouts();
  cancelTokenSource.cancel();

  isStreaming = false;

  buttonStart.textContent = 'Start';
  buttonFullScreen.disabled = true;

  if (guid) {
    await realtimeEnd();
  }
};

const handleButtonStartClick = () => {
  if (isStreaming) {
    stopStream();
  } else {
    startStream();
  }
};

const handleButtonFlipCameraClick = () => webcam.flip();

const activateFullscreen = (element) => {
  if (element.requestFullscreen) {
    element.requestFullscreen();
  } else if (element.mozRequestFullScreen) {
    element.mozRequestFullScreen();
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen();
  }
};

const deactivateFullscreen = () => {
  if (document.exitFullscreen) {
    document.exitFullscreen();
  } else if ((document as Document).mozCancelFullScreen) {
    (document as Document).mozCancelFullScreen();
  } else if ((document as Document).webkitExitFullscreen) {
    (document as Document).webkitExitFullscreen();
  }
};

const handleButtonFullscreenClick = () => {
  const fullscreenElement = document.fullscreenElement
    || (document as Document).mozFullScreenElement
    || (document as Document).webkitFullscreenElement;

  if (fullscreenElement) {
    deactivateFullscreen();
  } else {
    activateFullscreen(canvasLayers);
  }
};

const handleButtonDownloadClick = async () => {
  const bufferCanvas = document.createElement('canvas');
  const bufferContext = bufferCanvas.getContext('2d');

  bufferCanvas.width = canvasImage.width;
  bufferCanvas.height = canvasImage.height;
  bufferContext.drawImage(canvasImage, 0, 0);
  bufferContext.drawImage(canvasBoxes, 0, 0);

  const link = document.createElement('a');
  link.download = 'filename.png';
  link.href = bufferCanvas.toDataURL('image/jpeg', quality);
  link.click();
};

const handleInputFPSChange = async (evt) => {
  const input = evt.target as HTMLInputElement;

  if (Number(input.value) > MAX_FPS) {
    input.value = `${MAX_FPS}`;
  }

  localStorage.setItem('fps', input.value);
  fps = Number(input.value);

  if (isStreaming) {
    await stopStream();
    await startStream();
  }
};

const handleInputDelayChange = async (evt) => {
  const input = evt.target as HTMLInputElement;

  localStorage.setItem('delay', input.value);
  delay = Number(input.value);

  if (isStreaming) {
    await stopStream();
    await startStream();
  }
};

const handleInputQualityChange = async (evt) => {
  const input = evt.target as HTMLInputElement;

  if (Number(input.value) > MAX_QUALITY) {
    input.value = `${MAX_QUALITY}`;
  }

  localStorage.setItem('quality', input.value);
  quality = Number(input.value);

  if (isStreaming) {
    await stopStream();
    await startStream();
  }
};

buttonStart.addEventListener('click', handleButtonStartClick);
buttonFlipCamera.addEventListener('click', handleButtonFlipCameraClick);
buttonFullScreen.addEventListener('click', handleButtonFullscreenClick);
buttonDownload.addEventListener('click', handleButtonDownloadClick);
inputFps.addEventListener('input', handleInputFPSChange);
inputQuality.addEventListener('input', handleInputQualityChange);
inputDelay.addEventListener('input', handleInputDelayChange);
