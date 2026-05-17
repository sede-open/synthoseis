/**
 * generates a UID factory
 * @internal
 * @example
 * const uid = generateUID();
 * uid(object) = 1;
 * uid(object) = 1;
 * uid(anotherObject) = 2;
 */
export declare const generateUID: () => (item: any, index?: number) => string;
/**
 * @name uid
 * returns an UID associated with {item}
 * @param {Object} item - object to generate UID for
 * @param {Number} index, a fallback index
 * @example
 * uid(object) == 1;
 * uid(object) == 1;
 * uid(anotherObject) == 2;
 * uid("not object", 42) == 42
 *
 * @see {@link useUID}
 */
export declare const uid: (item: any, index?: number) => string;
