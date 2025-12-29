export interface Renderer {
    initialize(container: HTMLElement): void;
    resize(width: number, height: number): void;
    render(): void;
    destroy(): void;
}
