import * as React from 'react';
export declare type IdSourceType = {
    value: number;
    prefix: string;
    uid: (item: any, index?: number) => string;
};
export declare const createSource: (prefix?: string) => IdSourceType;
export interface UIDProps {
    name?: (n: string | number) => string;
    idSource?: IdSourceType;
    children: (id: string, uid: (item: any, index?: number) => string) => React.ReactNode;
}
export declare const counter: IdSourceType;
export declare const source: React.Context<IdSourceType>;
export declare const getId: (source: IdSourceType) => number;
export declare const getPrefix: (source?: IdSourceType) => string;
