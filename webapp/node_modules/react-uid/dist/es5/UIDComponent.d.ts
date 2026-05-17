import * as React from 'react';
import { UIDProps } from './context';
/**
 * @deprecated
 * UID in form of renderProps (not SSR friendly)
 * @see https://github.com/thearnica/react-uid#react-components
 * @example
 * // get UID to connect label to input
 * <UID>
 *   {(id)} => <label htmlFor={id}><input id={id}/>}
 * </UID>
 *
 * // get uid to generate uid for a keys in a list
 * <UID>
 *   {(, uid)} => items.map(item => <li key={uid(item) />)}
 * </UID>
 */
export declare class UID extends React.Component<UIDProps> {
    state: {
        quartz: import("./context").IdSourceType;
        prefix: string;
        id: number;
    };
    uid: (item: any) => string;
    render(): React.ReactNode;
}
