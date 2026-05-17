import { BBox } from 'geojson';
import { AllGeoJSON } from '@turf/helpers';

/**
 * Calculates the bounding box for any GeoJSON object, including FeatureCollection.
 * Uses geojson.bbox if available and options.recompute is not set.
 *
 * @function
 * @param {GeoJSON} geojson any GeoJSON object
 * @param {Object} [options={}] Optional parameters
 * @param {boolean} [options.recompute] Whether to ignore an existing bbox property on geojson
 * @returns {BBox} bbox extent in [minX, minY, maxX, maxY] order
 * @example
 * var line = turf.lineString([[-74, 40], [-78, 42], [-82, 35]]);
 * var bbox = turf.bbox(line);
 * var bboxPolygon = turf.bboxPolygon(bbox);
 *
 * //addToMap
 * var addToMap = [line, bboxPolygon]
 */
declare function bbox(geojson: AllGeoJSON, options?: {
    recompute?: boolean;
}): BBox;

export { bbox, bbox as default };
