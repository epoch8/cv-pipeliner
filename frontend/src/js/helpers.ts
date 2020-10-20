import { Box } from './interfaces';

const TEXT_BASELINE = 'bottom';
const TEXT_ALIGN = 'left';
const FONT_SIZE = 18;
const FONT = `${FONT_SIZE}px sans-serif`;
const HORIZONTAL_MARGIN = 4;
const VERTICAL_MARGIN = 4;
const BLACK_COLOR = '#000000';
const RED_COLOR = '#ff0000';
const labelSizes = {
  WIDTH: 45,
  HEIGHT: 15,
};

export const drawBox = (
  ctx: CanvasRenderingContext2D,
  box: Box,
  color = RED_COLOR,
  lineWidth = 3,
): void => {
  const [width, height] = [box.xmax - box.xmin, box.ymax - box.ymin];
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.rect(
    box.xmin + lineWidth,
    box.ymin + lineWidth,
    width - lineWidth * 2,
    height - lineWidth * 2,
  );
  ctx.stroke();
};

export const drawText = (
  ctx: CanvasRenderingContext2D,
  box: Box,
  text: string,
  color: string,
): void => {
  const [width, height] = [box.xmax - box.xmin, box.ymax - box.ymin];
  ctx.textBaseline = TEXT_BASELINE;
  ctx.font = FONT;
  ctx.textAlign = TEXT_ALIGN;
  const textWidth = ctx.measureText(text).width + 2 * HORIZONTAL_MARGIN;
  const textHeight = FONT_SIZE + 1 * VERTICAL_MARGIN;
  ctx.save();
  ctx.translate(box.xmin + width / 2, box.ymin);
  ctx.fillStyle = color;
  ctx.fillRect(-width / 2, -height, width, height);
  ctx.stroke();
  ctx.scale(width / textWidth, height / textHeight);
  ctx.translate(HORIZONTAL_MARGIN, -VERTICAL_MARGIN / 2);
  ctx.fillStyle = BLACK_COLOR;
  ctx.fillText(text, -textWidth / 2, 0);
  ctx.restore();
};

export const drawLabel = (
  ctx: CanvasRenderingContext2D,
  box: Box,
  text: string,
  color: string,
): void => {
  const { xmin, ymin } = box;
  ctx.beginPath();
  drawBox(
    ctx,
    {
      xmin,
      ymin,
      xmax: xmin + labelSizes.WIDTH,
      ymax: ymin - labelSizes.HEIGHT,
    } as Box,
    BLACK_COLOR,
    1,
  );
  drawText(
    ctx,
    {
      xmin,
      ymin,
      xmax: xmin + labelSizes.WIDTH,
      ymax: ymin + labelSizes.HEIGHT,
    } as Box,
    text,
    color,
  );
  ctx.closePath();
};
