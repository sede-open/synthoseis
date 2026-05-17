import { GeoJsonProperties, Feature, Point } from 'geojson';
import { AllGeoJSON } from '@turf/helpers';

/**
 * Computes the centroid as the mean of all vertices within the object.
 *
 * @function
 * @param {GeoJSON} geojson GeoJSON to be centered
 * @param {Object} [options={}] Optional Parameters
 * @param {Object} [options.properties={}] an Object that is used as the {@link Feature}'s properties
 * @returns {Feature<Point>} the centroid of the input object
 * @example
 * var polygon = turf.polygon([[[-81, 41], [-88, 36], [-84, 31], [-80, 33], [-77, 39], [-81, 41]]]);
 *
 * var centroid = turf.centroid(polygon);
 *
 * //addToMap
 * var addToMap = [polygon, centroid]
 */
declare function centroid<P extends GeoJsonProperties = GeoJsonProperties>(geojson: AllGeoJSON, options?: {
    properties?: P;
}): Feature<Point, P>;

export { centroid, centroid as default };
