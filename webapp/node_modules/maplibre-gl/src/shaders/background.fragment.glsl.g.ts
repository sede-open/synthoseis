// This file is generated. Edit build/generate-shaders.ts, then run `npm run codegen`.
export default 'uniform vec4 u_color;uniform float u_opacity;void main() {gl_FragColor=u_color*u_opacity;\n#ifdef OVERDRAW_INSPECTOR\ngl_FragColor=vec4(1.0);\n#endif\n}';
