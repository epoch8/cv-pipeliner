export interface Box {
  label: string;
  xmax: number;
  xmin: number;
  ymax: number;
  ymin: number;
}

export interface Document extends HTMLDocument {
  mozCancelFullScreen?: any;
  mozFullScreenElement?: any;
  msExitFullscreen?: any;
  msFullscreenElement?: any;
  webkitExitFullscreen?: any;
  webkitFullscreenElement?: any;
}
