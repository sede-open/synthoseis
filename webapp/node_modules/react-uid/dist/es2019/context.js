import * as React from 'react';
import { generateUID } from './uid';
export const createSource = (prefix = '') => ({
    value: 1,
    prefix: prefix,
    uid: generateUID(),
});
export const counter = createSource();
export const source = React.createContext(createSource());
export const getId = (source) => source.value++;
export const getPrefix = (source) => (source ? source.prefix : '');
