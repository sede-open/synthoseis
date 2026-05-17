/**
 * Actually a uid function - call with any argument
 * @returns {String}
 * @see {@link uid}
 */
declare type SeedGenerator = (id: any) => string;
/**
 * returns and unique id. SSR friendly
 * returns {String}
 * @see {@link UIDConsumer}
 * @see https://github.com/thearnica/react-uid#hooks-168
 * @example
 * const id = useUID();
 * id == 1; // for example
 */
export declare const useUID: () => string;
/**
 * returns an uid generator
 * @see {@link UIDConsumer}
 * @see https://github.com/thearnica/react-uid#hooks-168
 * @example
 * const uid = useUIDSeed();
 * return (
 *  <>
 *    <label for={seed('email')}>Email: </label>
 *    <input id={seed('email')} name="email" />
 *    {data.map(item => <div key={seed(item)}>...</div>
 *  </>
 * )
 */
export declare const useUIDSeed: () => SeedGenerator;
export {};
