import { mat4 } from 'gl-matrix';
import type { CubeRenderer } from './renderer';
import { asciiVertexShader, asciiFragmentShader, createAsciiTexture } from './ascii-tg-shader';

export class TypeGpuRenderer implements CubeRenderer {
    private canvas: HTMLCanvasElement | null = null;
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;

    // Pipelines
    private cubePipeline: any = null;
    private asciiPipeline: any = null;

    // Resources
    private renderTarget: GPUTexture | null = null;
    private asciiTexture: GPUTexture | null = null;
    private sampler: GPUSampler | null = null;
    private vertexBuffer: GPUBuffer | null = null;
    private faceTexture: GPUTexture | null = null;

    // Uniforms
    private cubeUniformBuffer: GPUBuffer | null = null;
    private asciiUniformBuffer: GPUBuffer | null = null;
    private quadBuffer: GPUBuffer | null = null;
    private depthTexture: GPUTexture | null = null;

    // Interaction
    private isDragging = false;
    private previousMousePosition = { x: 0, y: 0 };
    private targetRotation = { x: 0, y: 0 };
    private currentRotation = { x: 0, y: 0 };

    async initialize(container: HTMLElement): Promise<void> {
        console.log('Renderer: Requesting adapter...');
        const adapter = await navigator.gpu?.requestAdapter();
        console.log('Renderer: Adapter found:', !!adapter);

        console.log('Renderer: Requesting device...');
        this.device = await adapter?.requestDevice() ?? null;
        console.log('Renderer: Device found:', !!this.device);

        if (!this.device) throw new Error('WebGPU not supported');

        this.canvas = document.createElement('canvas');
        this.canvas.width = container.clientWidth * window.devicePixelRatio;
        this.canvas.height = container.clientHeight * window.devicePixelRatio;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        container.appendChild(this.canvas);

        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied',
        });

        this.setupCubeResources();
        this.setupAsciiResources();
        this.setupInteraction(container);

        console.log('Renderer: Loading face texture...');
        // await this.loadFaceTexture('/MetaMask-icon-fox-developer-inverted.jpg')
        await this.loadFaceTexture('/MetaMask-icon-fox.jpg')
        // await this.loadFaceTexture('/mona.jpg')

        // Initial resize to setup the render target and viewport
        this.resize(container.clientWidth, container.clientHeight);
    }

    private setupCubeResources(): void {
        if (!this.device) return;

        // Cube vertices (Pos + UV)
        const vertices = new Float32Array([
            // Front face
            -0.8, -0.8, 0.8, 0, 1,
            0.8, -0.8, 0.8, 1, 1,
            0.8, 0.8, 0.8, 1, 0,
            -0.8, -0.8, 0.8, 0, 1,
            0.8, 0.8, 0.8, 1, 0,
            -0.8, 0.8, 0.8, 0, 0,

            // Back face
            -0.8, -0.8, -0.8, 1, 1,
            -0.8, 0.8, -0.8, 1, 0,
            0.8, 0.8, -0.8, 0, 0,
            -0.8, -0.8, -0.8, 1, 1,
            0.8, 0.8, -0.8, 0, 0,
            0.8, -0.8, -0.8, 0, 1,

            // Top
            -0.8, 0.8, -0.8, 0, 0,
            -0.8, 0.8, 0.8, 0, 1,
            0.8, 0.8, 0.8, 1, 1,
            -0.8, 0.8, -0.8, 0, 0,
            0.8, 0.8, 0.8, 1, 1,
            0.8, 0.8, -0.8, 1, 0,

            // Bottom
            -0.8, -0.8, -0.8, 0, 1,
            0.8, -0.8, -0.8, 1, 1,
            0.8, -0.8, 0.8, 1, 0,
            -0.8, -0.8, -0.8, 0, 1,
            0.8, -0.8, 0.8, 1, 0,
            -0.8, -0.8, 0.8, 0, 0,

            // Right
            0.8, -0.8, -0.8, 1, 1,
            0.8, 0.8, -0.8, 1, 0,
            0.8, 0.8, 0.8, 0, 0,
            0.8, -0.8, -0.8, 1, 1,
            0.8, 0.8, 0.8, 0, 0,
            0.8, -0.8, 0.8, 0, 1,

            // Left
            -0.8, -0.8, -0.8, 0, 1,
            -0.8, -0.8, 0.8, 1, 1,
            -0.8, 0.8, 0.8, 1, 0,
            -0.8, -0.8, -0.8, 0, 1,
            -0.8, 0.8, 0.8, 1, 0,
            -0.8, 0.8, -0.8, 0, 0,
        ]);

        this.vertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
        this.vertexBuffer.unmap();

        this.cubeUniformBuffer = this.device.createBuffer({
            size: 64, // 4x4 matrix
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Simple cube pipeline (would normally use TypeGPU layouts)
        // For brevity, using standard WebGPU calls where TypeGPU abstractions are not strictly needed
        this.cubePipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: `
                        @group(0) @binding(0) var<uniform> mvp: mat4x4f;
                        struct VertexOutput {
                            @builtin(position) pos: vec4f,
                            @location(0) uv: vec2f,
                        }
                        @vertex
                        fn main(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VertexOutput {
                            return VertexOutput(mvp * vec4f(pos, 1.0), uv);
                        }
                    `
                }),
                entryPoint: 'main',
                buffers: [{
                    arrayStride: 20,
                    attributes: [
                        { format: 'float32x3', offset: 0, shaderLocation: 0 },
                        { format: 'float32x2', offset: 12, shaderLocation: 1 },
                    ]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: `
                        @group(0) @binding(1) var t: texture_2d<f32>;
                        @group(0) @binding(2) var s: sampler;
                        @fragment
                        fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
                            return textureSample(t, s, uv);
                        }
                    `
                }),
                entryPoint: 'main',
                targets: [{ format: 'rgba8unorm' }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
        });
    }

    private setupAsciiResources(): void {
        if (!this.device || !this.canvas) return;

        this.sampler = this.device.createSampler({
            minFilter: 'linear',
            magFilter: 'linear',
        });

        this.asciiTexture = createAsciiTexture(this.device);

        this.asciiUniformBuffer = this.device.createBuffer({
            size: 80, // size of AsciiParams
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.asciiPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: asciiVertexShader }),
                entryPoint: 'main',
                buffers: [{
                    arrayStride: 8,
                    attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 0 }]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({ code: asciiFragmentShader }),
                entryPoint: 'main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-strip' }
        });

        // Fullscreen quad buffer
        this.quadBuffer = this.device.createBuffer({
            size: 4 * 2 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.quadBuffer.getMappedRange()).set([
            -1, -1, 1, -1, -1, 1, 1, 1
        ]);
        this.quadBuffer.unmap();

        this.updateAsciiUniforms();
    }

    private async loadFaceTexture(url: string): Promise<void> {
        if (!this.device) return;

        const image = await new Promise<HTMLImageElement>((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
            img.src = url;
        });

        const bitmap = await createImageBitmap(image, {
            resizeWidth: 512,
            resizeHeight: 512,
            resizeQuality: 'high',
        });

        this.faceTexture = this.device.createTexture({
            size: [bitmap.width, bitmap.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.faceTexture },
            [bitmap.width, bitmap.height]
        );
    }

    private updateAsciiUniforms(): void {
        if (!this.device || !this.canvas || !this.asciiUniformBuffer) return;

        const data = new Float32Array([
            9.0, // uCharCount (based on CHAR_SET.length)
            6.0 * window.devicePixelRatio, // uFontSize
            this.canvas.width, this.canvas.height, // uResolution
            1.0, 1, 1.0// uColor (white)
        ]);
        this.device.queue.writeBuffer(this.asciiUniformBuffer, 0, data);
    }

    private setupInteraction(container: HTMLElement): void {
        container.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.previousMousePosition = { x: e.clientX, y: e.clientY };
            container.style.cursor = 'grabbing';
        });

        container.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaMove = {
                    x: e.clientX - this.previousMousePosition.x,
                    y: e.clientY - this.previousMousePosition.y
                };
                this.targetRotation.x += deltaMove.y * 0.01;
                this.targetRotation.y += deltaMove.x * 0.01;
                this.previousMousePosition = { x: e.clientX, y: e.clientY };
            }
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            container.style.cursor = 'grab';
        });

        window.addEventListener('keydown', (e) => {
            const SPEED = 0.1;
            if (e.key === 'ArrowUp') this.targetRotation.x -= SPEED;
            if (e.key === 'ArrowDown') this.targetRotation.x += SPEED;
            if (e.key === 'ArrowLeft') this.targetRotation.y -= SPEED;
            if (e.key === 'ArrowRight') this.targetRotation.y += SPEED;
        });
    }

    resize(width: number, height: number): void {
        if (!this.canvas || !this.device) return;
        this.canvas.width = width * window.devicePixelRatio;
        this.canvas.height = height * window.devicePixelRatio;

        // Recreation of renderTarget
        this.renderTarget?.destroy();
        this.renderTarget = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Recreation of depthTexture
        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.updateAsciiUniforms();
    }

    render(): void {
        if (!this.device || !this.context || !this.canvas || !this.cubePipeline || !this.faceTexture || !this.sampler || !this.asciiPipeline || !this.renderTarget || !this.asciiTexture) return;

        // Smooth rotation
        this.currentRotation.x += (this.targetRotation.x - this.currentRotation.x) * 0.1;
        this.currentRotation.y += (this.targetRotation.y - this.currentRotation.y) * 0.1;

        if (!this.isDragging) {
            this.targetRotation.y += 0.002;
            this.targetRotation.x += 0.001;
        }

        // MVP Matrix
        const projection = mat4.perspective(mat4.create(), Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 100);
        const view = mat4.lookAt(mat4.create(), [0, 0, 4.5], [0, 0, 0], [0, 1, 0]);
        const model = mat4.create();
        mat4.rotateX(model, model, this.currentRotation.x);
        mat4.rotateY(model, model, this.currentRotation.y);

        const mvp = mat4.multiply(mat4.create(), projection, mat4.multiply(mat4.create(), view, model));
        this.device.queue.writeBuffer(this.cubeUniformBuffer!, 0, (mvp as unknown as Float32Array).buffer as ArrayBuffer);

        const commandEncoder = this.device.createCommandEncoder();

        // Pass 1: Render Cube to renderTarget
        const cubePass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.renderTarget.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        });

        const cubeBindGroup = this.device.createBindGroup({
            layout: this.cubePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cubeUniformBuffer! } },
                { binding: 1, resource: this.faceTexture.createView() },
                { binding: 2, resource: this.sampler },
            ]
        });

        cubePass.setPipeline(this.cubePipeline);
        cubePass.setBindGroup(0, cubeBindGroup);
        cubePass.setVertexBuffer(0, this.vertexBuffer!);
        cubePass.draw(36);
        cubePass.end();

        // Pass 2: ASCII Post-processing to Screen
        const asciiPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        const asciiBindGroup = this.device.createBindGroup({
            layout: this.asciiPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.renderTarget.createView() },
                { binding: 1, resource: this.asciiTexture.createView() },
                { binding: 2, resource: this.sampler },
                { binding: 3, resource: { buffer: this.asciiUniformBuffer! } },
            ]
        });

        asciiPass.setPipeline(this.asciiPipeline);
        asciiPass.setBindGroup(0, asciiBindGroup);
        asciiPass.setVertexBuffer(0, this.quadBuffer!);
        asciiPass.draw(4);
        asciiPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    destroy(): void {
        this.renderTarget?.destroy();
        this.depthTexture?.destroy();
        this.faceTexture?.destroy();
        this.asciiTexture?.destroy();
        this.vertexBuffer?.destroy();
        this.cubeUniformBuffer?.destroy();
        this.asciiUniformBuffer?.destroy();
        this.quadBuffer?.destroy();
    }
}
