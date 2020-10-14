import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

import { drawBox, drawLabel } from './helpers';
import { BOX_COLORS, Url } from './constants';

interface VideoConstraints {
  deviceId?: {
    exact: string;
  };
  facingMode?: VideoFacingModeEnum;
}

interface SnapProps {
  quality?: number;
  type?: string;
}

export default class Webcam {
  private readonly _canvasElement: HTMLCanvasElement | null;

  private _colorCache: { [key: string]: string };

  private _facingMode: VideoFacingModeEnum;

  private _guid: number;

  private _selectedDeviceId: string;

  private _streamList: MediaStream[];

  private _webcamList: MediaDeviceInfo[];

  private readonly _webcamElement: HTMLVideoElement;

  constructor(
    webcamElement: HTMLVideoElement,
    facingMode: VideoFacingModeEnum = 'user',
    canvasElement: HTMLCanvasElement | null = null,
  ) {
    this._canvasElement = canvasElement;
    this._colorCache = {};
    this._facingMode = facingMode;
    this._guid = null;
    this._selectedDeviceId = '';
    this._streamList = [];
    this._webcamElement = webcamElement;
    this._webcamElement.height = this._webcamElement.height || this._webcamElement.width * (3 / 4);
    this._webcamElement.width = this._webcamElement.width || 640;
    this._webcamList = [];
  }

  get facingMode(): VideoFacingModeEnum {
    return this._facingMode;
  }

  set facingMode(value: VideoFacingModeEnum) {
    this._facingMode = value;
  }

  get webcamList(): MediaDeviceInfo[] {
    return this._webcamList;
  }

  get webcamCount(): number {
    return this._webcamList.length;
  }

  get selectedDeviceId(): string {
    return this._selectedDeviceId;
  }

  getVideoInputs(mediaDevices: MediaDeviceInfo[]): MediaDeviceInfo[] {
    this._webcamList = [];

    mediaDevices.forEach((mediaDevice) => {
      if (mediaDevice.kind === 'videoinput') {
        this._webcamList.push(mediaDevice);
      }
    });

    if (this._webcamList.length === 1) {
      this._facingMode = 'user';
    }

    return this._webcamList;
  }

  getMediaConstraints(): { video: VideoConstraints, audio: boolean } {
    const videoConstraints: VideoConstraints = {};

    if (this._selectedDeviceId === '') {
      videoConstraints.facingMode = this._facingMode;
    } else {
      videoConstraints.deviceId = { exact: this._selectedDeviceId };
    }

    return {
      video: videoConstraints,
      audio: false,
    };
  }

  selectCamera(): void {
    // eslint-disable-next-line no-restricted-syntax
    for (const webcam of this._webcamList) {
      const front = this._facingMode === 'user' && webcam.label.toLowerCase().includes('front');
      const back = this._facingMode === 'environment' && webcam.label.toLowerCase().includes('back');

      if (front || back) {
        this._selectedDeviceId = webcam.deviceId;
        break;
      }
    }
  }

  flip(): void {
    this._facingMode = this._facingMode === 'user' ? 'environment' : 'user';
    this._webcamElement.style.transform = '';
    this.selectCamera();
  }

  async start(startStream = true): Promise<VideoFacingModeEnum | string> {
    return new Promise((resolve, reject) => {
      (async () => {
        try {
          await this.stop();

          const mediaConstrains = this.getMediaConstraints();
          const stream = await navigator.mediaDevices.getUserMedia(mediaConstrains);
          this._streamList.push(stream);

          await this.info();
          this.selectCamera();

          if (startStream) {
            await this.stream();
            resolve(this._facingMode);
          } else {
            resolve(this._selectedDeviceId);
          }
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  async info(): Promise<MediaDeviceInfo[]> {
    return new Promise((resolve, reject) => {
      (async () => {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          this.getVideoInputs(devices);
          resolve(this._webcamList);
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  async stream(): Promise<VideoFacingModeEnum> {
    return new Promise((resolve, reject) => {
      (async () => {
        try {
          const mediaConstraints = this.getMediaConstraints();
          const stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
          this._streamList.push(stream);
          this._webcamElement.srcObject = stream;

          if (this._facingMode === 'user') {
            this._webcamElement.style.transform = 'scale(-1,1)';
          }

          this._webcamElement.play();
          resolve(this._facingMode);
        } catch (error) {
          console.log(error);
          reject(error);
        }
      })();
    });
  }

  async stop(): Promise<void> {
    this._streamList.forEach((stream) => {
      stream.getTracks().forEach((track) => track.stop());
    });
  }

  snap({ type = 'image/jpeg', quality = 0.92 }: SnapProps): Promise<string> {
    if (this._canvasElement !== null) {
      this._canvasElement.height = this._webcamElement.scrollHeight;
      this._canvasElement.width = this._webcamElement.scrollWidth;

      const context = this._canvasElement.getContext('2d');

      if (!context) {
        throw new Error('context is missing');
      }

      context.clearRect(
        0,
        0,
        this._canvasElement.width,
        this._canvasElement.height,
      );

      context.drawImage(
        this._webcamElement,
        0,
        0,
        this._canvasElement.width,
        this._canvasElement.height,
      );

      return new Promise((resolve, reject) => {
        const formData = new FormData();
        this._canvasElement.toBlob(async (blob) => {
          try {
            formData.append('image', blob);
            const { data: { bboxes } } = await axios.post(`${Url.REALTIME_PREDICT}/${this._guid}`, formData);

            if (!bboxes.length) {
              reject();
            }

            bboxes.forEach((box, index) => {
              if (!this._colorCache[box.label]) {
                this._colorCache[box.label] = BOX_COLORS[index];
              }

              drawBox(context, box, this._colorCache[box.label]);
              drawBox(context, box, 'black', 1);
              drawLabel(context, box, box.label, this._colorCache[box.label]);
            });

            const url = this._canvasElement.toDataURL(type, quality);
            resolve(url);
          } catch (err) {
            reject(err);
          }
        });
      });
    }

    throw new Error('canvas element is missing');
  }

  async realtimeStart(fps: number, detectionDelay: number) {
    try {
      if (this._guid) {
        await this.realtimeEnd();
      }

      this._guid = uuidv4();
      const formData = new FormData();
      formData.append('fps', `${fps}`);
      formData.append('detection_delay', `${detectionDelay}`);
      await axios.post(`${Url.REALTIME_START}/${this._guid}`, formData);
    } catch (err) {
      console.log('realtimeStart', err);
    }
  }

  async realtimeEnd() {
    try {
      await axios.post(`${Url.REALTIME_END}/${this._guid}`);
      this._guid = null;
    } catch (err) {
      console.log('realtimeEnd', err);
    }
  }
}
