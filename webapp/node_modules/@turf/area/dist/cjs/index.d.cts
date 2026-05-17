import { Feature, FeatureCollection, Geometry } from 'geojson';

/**
 * Calculates the geodesic area in square meters of one or more polygons.
 *
 * @function
 * @param {GeoJSON} geojson input polygon(s) as {@link Geometry}, {@link Feature}, or {@link FeatureCollection}
 * @returns {number} area in square meters
 * @example
 * var polygon = turf.polygon([[[125, -15], [113, -22], [154, -27], [144, -15], [125, -15]]]);
 *
 * var area = turf.area(polygon);
 *
 * //addToMap
 * var addToMap = [polygon]
 * polygon.properties.area = area
 */
declare function area(geojson: Feature<any> | FeatureCollection<any> | Geometry): number;

export { area, area as default };
