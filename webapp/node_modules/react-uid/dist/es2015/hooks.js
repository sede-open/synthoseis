import * as React from 'react';
import { useContext, useState } from 'react';
import { counter, getId, getPrefix, source } from './context';
var generateUID = function (context) {
    var quartz = context || counter;
    var prefix = getPrefix(quartz);
    var id = getId(quartz);
    var uid = prefix + id;
    var gen = function (item) { return uid + quartz.uid(item); };
    return { uid: uid, gen: gen };
};
var useUIDState = function () {
    if (process.env.NODE_ENV !== 'production') {
        if (!('useContext' in React)) {
            throw new Error('Hooks API requires React 16.8+');
        }
    }
    var context = useContext(source);
    var uid = useState(function () { return generateUID(context); })[0];
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
export var useUID = function () {
    var uid = useUIDState().uid;
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
export var useUIDSeed = function () {
    var gen = useUIDState().gen;
    return gen;
};
