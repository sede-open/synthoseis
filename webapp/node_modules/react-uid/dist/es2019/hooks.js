import * as React from 'react';
import { useContext, useState } from 'react';
import { counter, getId, getPrefix, source } from './context';
const generateUID = (context) => {
    const quartz = context || counter;
    const prefix = getPrefix(quartz);
    const id = getId(quartz);
    const uid = prefix + id;
    const gen = (item) => uid + quartz.uid(item);
    return { uid, gen };
};
const useUIDState = () => {
    if (process.env.NODE_ENV !== 'production') {
        if (!('useContext' in React)) {
            throw new Error('Hooks API requires React 16.8+');
        }
    }
    const context = useContext(source);
    const [uid] = useState(() => generateUID(context));
    return uid;
};
/**
 * returns and unique id. SSR friendly
 * returns {String}
 * @see {@link UIDConsumer}
 * @see https://github.com/thearnica/react-uid#hooks-168
 * @example
 * const id = useUID();
 * id == 1; // for example
 */
export const useUID = () => {
    const { uid } = useUIDState();
    return uid;
};
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
export const useUIDSeed = () => {
    const { gen } = useUIDState();
    return gen;
};
