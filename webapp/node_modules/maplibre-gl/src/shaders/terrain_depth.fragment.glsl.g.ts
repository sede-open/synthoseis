// This file is generated. Edit build/generate-shaders.ts, then run `npm run codegen`.
export default 'varying float v_depth;const highp vec4 bitSh=vec4(256.*256.*256.,256.*256.,256.,1.);const highp vec4 bitMsk=vec4(0.,vec3(1./256.0));highp vec4 pack(highp float value) {highp vec4 comp=fract(value*bitSh);comp-=comp.xxyz*bitMsk;return comp;}void main() {gl_FragColor=pack(v_depth);}';
