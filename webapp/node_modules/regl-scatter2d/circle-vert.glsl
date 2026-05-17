precision highp float;

attribute float x, y, xFract, yFract;
attribute float size, borderSize;
attribute vec4 colorId, borderColorId;
attribute float isActive;

// `invariant` effectively turns off optimizations for the position.
// We need this because -fast-math on M1 Macs is re-ordering
// floating point operations in a way that causes floating point
// precision limits to put points in the wrong locations.
invariant gl_Position;

uniform bool constPointSize;
uniform float pixelRatio;
uniform vec2 paletteSize, scale, scaleFract, translate, translateFract;
uniform sampler2D paletteTexture;

const float maxSize = 100.;

varying vec4 fragColor, fragBorderColor;
varying float fragBorderRadius, fragWidth;

float pointSizeScale = (constPointSize) ? 2. : pixelRatio;

bool isDirect = (paletteSize.x < 1.);

vec4 getColor(vec4 id) {
  return isDirect ? id / 255. : texture2D(paletteTexture,
    vec2(
      (id.x + .5) / paletteSize.x,
      (id.y + .5) / paletteSize.y
    )
  );
}

void main() {
  // ignore inactive points
  if (isActive == 0.) return;

  vec2 position = vec2(x, y);
  vec2 positionFract = vec2(xFract, yFract);

  vec4 color = getColor(colorId);
  vec4 borderColor = getColor(borderColorId);

  float size = size * maxSize / 255.;
  float borderSize = borderSize * maxSize / 255.;

  gl_PointSize = (size + borderSize) * pointSizeScale;

  vec2 pos = (position + translate) * scale
      + (positionFract + translateFract) * scale
      + (position + translate) * scaleFract
      + (positionFract + translateFract) * scaleFract;

  gl_Position = vec4(pos * 2. - 1., 0., 1.);

  fragBorderRadius = 1. - 2. * borderSize / (size + borderSize);
  fragColor = color;
  fragBorderColor = borderColor.a == 0. || borderSize == 0. ? vec4(color.rgb, 0.) : borderColor;
  fragWidth = 1. / gl_PointSize;
}
