"use strict";Object.defineProperty(exports, "__esModule", {value: true});// index.ts
var _helpers = require('@turf/helpers');
var _meta = require('@turf/meta');
function centroid(geojson, options = {}) {
  let xSum = 0;
  let ySum = 0;
  let len = 0;
  _meta.coordEach.call(void 0, 
    geojson,
    function(coord) {
      xSum += coord[0];
      ySum += coord[1];
      len++;
    },
    true
  );
  return _helpers.point.call(void 0, [xSum / len, ySum / len], options.properties);
}
var index_default = centroid;



exports.centroid = centroid; exports.default = index_default;
//# sourceMappingURL=index.cjs.map