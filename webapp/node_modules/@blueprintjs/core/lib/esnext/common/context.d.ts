export type Validator<K> = (props: {
    [key: string]: any;
}, propName: K, componentName: string, location: string, propFullName: string) => Error | undefined;
export type ValidationMap<T> = {
    [K in keyof T]?: Validator<K>;
};
