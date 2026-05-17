// This file is generated. Edit build/generate-shaders.ts, then run `npm run codegen`.
export default 'attribute vec2 a_pos;uniform mat4 u_matrix;\n#pragma mapbox: define highp vec4 color\n#pragma mapbox: define lowp float opacity\nvoid main() {\n#pragma mapbox: initialize highp vec4 color\n#pragma mapbox: initialize lowp float opacity\ngl_Position=u_matrix*vec4(a_pos,0,1);}';
