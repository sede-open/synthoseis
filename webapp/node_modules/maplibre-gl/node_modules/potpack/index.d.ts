/**
 * @typedef {Object} PotpackBox
 * @property {number} w Box width.
 * @property {number} h Box height.
 * @property {number} [x] X coordinate in the resulting container.
 * @property {number} [y] Y coordinate in the resulting container.
 */
/**
 * @typedef {Object} PotpackStats
 * @property {number} w Width of the resulting container.
 * @property {number} h Height of the resulting container.
 * @property {number} fill The space utilization value (0 to 1). Higher is better.
 */
/**
 * Packs 2D rectangles into a near-square container.
 *
 * Mutates the {@link boxes} array: it's sorted (by height/width),
 * and box objects are augmented with `x`, `y` coordinates.
 *
 * @param {PotpackBox[]} boxes
 * @return {PotpackStats}
 */
export default function potpack(boxes: PotpackBox[]): PotpackStats;
export type PotpackBox = {
    /**
     * Box width.
     */
    w: number;
    /**
     * Box height.
     */
    h: number;
    /**
     * X coordinate in the resulting container.
     */
    x?: number | undefined;
    /**
     * Y coordinate in the resulting container.
     */
    y?: number | undefined;
};
export type PotpackStats = {
    /**
     * Width of the resulting container.
     */
    w: number;
    /**
     * Height of the resulting container.
     */
    h: number;
    /**
     * The space utilization value (0 to 1). Higher is better.
     */
    fill: number;
};
