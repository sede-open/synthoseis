// This file is generated. Edit build/generate-shaders.ts, then run `npm run codegen`.
export default 'varying vec4 v_color;void main() {gl_FragColor=v_color;\n#ifdef OVERDRAW_INSPECTOR\ngl_FragColor=vec4(1.0);\n#endif\n}';
