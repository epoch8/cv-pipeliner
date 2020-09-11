interface VideoConstraints {
  facingMode?: VideoFacingModeEnum;
  deviceId?: {
    exact: string;
  };
}

interface SnapProps {
  type?: string;
  quality?: number;
}

export default class Webcam {
  private readonly _canvasElement: HTMLCanvasElement | null;

  private _facingMode: VideoFacingModeEnum;

  private _selectedDeviceId: string;

  private readonly _snapSoundElement: HTMLAudioElement | null;

  private _streamList: MediaStream[];

  private _webcamList: MediaDeviceInfo[];

  private readonly _webcamElement: HTMLVideoElement;

  constructor(
    webcamElement: HTMLVideoElement,
    facingMode: VideoFacingModeEnum = 'user',
    canvasElement: HTMLCanvasElement | null = null,
    snapSoundElement: HTMLAudioElement | null = null,
  ) {
    this._canvasElement = canvasElement;
    this._facingMode = facingMode;
    this._selectedDeviceId = '';
    this._snapSoundElement = snapSoundElement;
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
          this.stop();
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

  stop(): void {
    this._streamList.forEach((stream) => {
      stream.getTracks().forEach((track) => track.stop());
    });
  }

  snap({ type = 'image/jpeg', quality = 0.92 }: SnapProps): string {
    if (this._canvasElement !== null) {
      if (this._snapSoundElement !== null) {
        this._snapSoundElement.play();
      }

      this._canvasElement.height = this._webcamElement.scrollHeight;
      this._canvasElement.width = this._webcamElement.scrollWidth;

      const context = this._canvasElement.getContext('2d');

      if (!context) {
        throw new Error('context is missing');
      }

      if (this._facingMode === 'user') {
        context.translate(this._canvasElement.width, 0);
        context.scale(-1, 1);
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

      return this._canvasElement.toDataURL(type, quality);
    }

    throw new Error('canvas element is missing');
  }
}
