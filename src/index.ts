// Web server for Image and Video Generation interface using Vertex AI Imagen and Veo APIs
import dotenv from 'dotenv';
import express, { Request, Response } from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { GoogleAuth } from 'google-auth-library';

// Load environment variables
dotenv.config();

// ============================================================================
// Types & Interfaces
// ============================================================================

interface ImageGenerationResult {
    id: number;
    prompts: string[]; // All prompts used for this image (original + edits)
    videoPrompts?: string[]; // Video generation prompts
    resultImages: string[];
    resultVideos?: string[]; // Videos generated from this image
}

interface PredictionResponse {
    mimeType: string;
    bytesBase64Encoded: string;
    prompt?: string; // Enhanced prompt if prompt rewriting is enabled
}

interface APIResponse {
    predictions?: Array<{
        mimeType?: string;
        bytesBase64Encoded?: string;
        prompt?: string;
    }>;
}

interface ImageMetadata {
    prompts: string[]; // Array of all prompts: [original, edit1, edit2, ...]
    enhancedPrompt?: string;
    videoPrompts?: string[]; // Video generation prompts
}

interface GenerateImageRequest {
    prompt: string;
    sampleCount?: number;
    aspectRatio?: string;
    safetySetting?: string;
    personGeneration?: string;
}

interface EditImageRequest {
    imagePath: string;
    editPrompt: string;
    sampleCount?: number;
    safetySetting?: string;
    personGeneration?: string;
}

interface GenerateVideoRequest {
    imagePath: string;
    prompt: string;
    aspectRatio?: string;
    duration?: number;
}

// ============================================================================
// Constants
// ============================================================================

const CONFIG = {
    PROJECT_ID: process.env.PROJECT_ID,
    LOCATION: process.env.LOCATION || 'us-central1',
    PORT: Number(process.env.PORT) || 3000,
    IMAGEN_MODEL: process.env.IMAGEN_MODEL || 'imagen-3.0-generate-001',
    IMAGEN_EDIT_MODEL: process.env.IMAGEN_EDIT_MODEL || 'imagen-3.0-capability-001',
    VEO_MODEL: process.env.VEO_MODEL || 'veo-3.0-generate-001',
} as const;

const PATHS = {
    SERVICE_ACCOUNT_KEY: path.resolve(process.cwd(), 'service-account-key.json'),
    OUTPUTS: path.join(process.cwd(), 'outputs'),
    METADATA: path.join(process.cwd(), 'outputs', 'metadata.json'),
} as const;

const CONSTRAINTS = {
    MIN_SAMPLE_COUNT: 1,
    MAX_SAMPLE_COUNT: 4,
    DEFAULT_SAMPLE_COUNT: 1,
    DEFAULT_ASPECT_RATIO: '1:1',
    DEFAULT_SAFETY_SETTING: 'block_medium_and_above',
    DEFAULT_PERSON_GENERATION: 'allow_adult',
} as const;

// ============================================================================
// Utility Functions
// ============================================================================

class FileUtils {
    static ensureDirectoryExists(dirPath: string): void {
        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }
    }

    static imageToBase64(filePath: string): string {
        if (!fs.existsSync(filePath)) {
            throw new Error(`File not found at path: ${filePath}`);
        }
        const fileBuffer = fs.readFileSync(filePath);
        return fileBuffer.toString('base64');
    }

    static extractGenerationId(filename: string): number | null {
        // Match result-{id} or result-{id}-{suffix} patterns
        // Examples: result-123.png, result-123-video.mp4
        const match = filename.match(/^result-(\d+)(?:-.*)?\./);
        return match ? parseInt(match[1], 10) : null;
    }

    static getFileExtension(mimeType: string): string {
        return mimeType.split('/')[1] || 'png';
    }
}

class ValidationUtils {
    static validatePrompt(prompt: unknown): string {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
            throw new Error('Prompt is required and must be a non-empty string');
        }
        return prompt.trim();
    }

    static validateImagePath(imagePath: unknown): string {
        if (!imagePath || typeof imagePath !== 'string') {
            throw new Error('Image path is required');
        }
        return imagePath;
    }

    static validateSampleCount(sampleCount: unknown): number {
        const count = parseInt(String(sampleCount || CONSTRAINTS.DEFAULT_SAMPLE_COUNT), 10);
        return Math.max(
            CONSTRAINTS.MIN_SAMPLE_COUNT,
            Math.min(CONSTRAINTS.MAX_SAMPLE_COUNT, count)
        );
    }
}

// ============================================================================
// Services
// ============================================================================

class AuthenticationService {
    static async getAccessToken(): Promise<string> {
        const auth = new GoogleAuth({
            keyFilename: PATHS.SERVICE_ACCOUNT_KEY,
            scopes: ['https://www.googleapis.com/auth/cloud-platform'],
        });
        const client = await auth.getClient();
        const accessToken = await client.getAccessToken();

        if (!accessToken.token) {
            throw new Error('Failed to obtain access token');
        }

        return accessToken.token;
    }
}

abstract class BaseAPIService {
    protected static validateConfig(): void {
        if (!CONFIG.PROJECT_ID || !CONFIG.LOCATION) {
            throw new Error('PROJECT_ID and LOCATION must be set in environment variables');
        }
    }

    protected static buildBaseUrl(model: string): string {
        this.validateConfig();
        return `https://${CONFIG.LOCATION}-aiplatform.googleapis.com/v1/projects/${CONFIG.PROJECT_ID}/locations/${CONFIG.LOCATION}/publishers/google/models/${model}`;
    }

    protected static buildApiEndpoint(model: string, suffix: string): string {
        return `${this.buildBaseUrl(model)}:${suffix}`;
    }

    protected static async makeApiRequest<T>(
        endpoint: string,
        requestBody: unknown,
        apiName: string = 'API'
    ): Promise<T> {
        const accessToken = await AuthenticationService.getAccessToken();

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${accessToken}`,
                'Content-Type': 'application/json; charset=utf-8',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
                `${apiName} API error: ${response.status} ${response.statusText}. ${errorText}`
            );
        }

        return (await response.json()) as T;
    }
}

class ImageGenerationService extends BaseAPIService {
    private static buildPredictEndpoint(model: string): string {
        return super.buildApiEndpoint(model, 'predict');
    }

    private static async makeImagenApiRequest(
        endpoint: string,
        requestBody: unknown
    ): Promise<APIResponse> {
        const responseData = await super.makeApiRequest<APIResponse>(
            endpoint,
            requestBody,
            'Imagen'
        );

        if (!responseData.predictions || responseData.predictions.length === 0) {
            throw new Error('Imagen API returned no predictions');
        }

        return responseData;
    }

    static async generateImage(
        prompt: string,
        sampleCount: number = CONSTRAINTS.DEFAULT_SAMPLE_COUNT,
        aspectRatio: string = CONSTRAINTS.DEFAULT_ASPECT_RATIO,
        safetySetting: string = CONSTRAINTS.DEFAULT_SAFETY_SETTING,
        personGeneration: string = CONSTRAINTS.DEFAULT_PERSON_GENERATION
    ): Promise<PredictionResponse[]> {
        const apiEndpoint = this.buildPredictEndpoint(CONFIG.IMAGEN_MODEL);

        const requestBody = {
            instances: [{ prompt }],
            parameters: {
                sampleCount: ValidationUtils.validateSampleCount(sampleCount),
                aspectRatio,
                safetySetting,
                personGeneration,
            },
        };

        const responseData = await this.makeImagenApiRequest(apiEndpoint, requestBody);

        return responseData.predictions!.map((pred) => ({
            mimeType: pred.mimeType || 'image/png',
            bytesBase64Encoded: pred.bytesBase64Encoded || '',
            prompt: pred.prompt,
        }));
    }

    static async editImage(
        imageBase64: string,
        editPrompt: string,
        sampleCount: number = CONSTRAINTS.DEFAULT_SAMPLE_COUNT,
        safetySetting: string = CONSTRAINTS.DEFAULT_SAFETY_SETTING,
        personGeneration: string = CONSTRAINTS.DEFAULT_PERSON_GENERATION
    ): Promise<PredictionResponse[]> {
        const apiEndpoint = this.buildPredictEndpoint(CONFIG.IMAGEN_EDIT_MODEL);

        const requestBody = {
            instances: [
                {
                    referenceImages: [
                        {
                            referenceType: 'REFERENCE_TYPE_RAW',
                            referenceId: 1,
                            referenceImage: {
                                bytesBase64Encoded: imageBase64,
                            },
                        },
                    ],
                    prompt: editPrompt,
                },
            ],
            parameters: {
                sampleCount: ValidationUtils.validateSampleCount(sampleCount),
                safetySetting,
                personGeneration,
            },
        };

        const responseData = await this.makeImagenApiRequest(apiEndpoint, requestBody);

        return responseData.predictions!.map((pred) => ({
            mimeType: pred.mimeType || 'image/png',
            bytesBase64Encoded: pred.bytesBase64Encoded || '',
            prompt: pred.prompt,
        }));
    }
}

class VideoGenerationService extends BaseAPIService {
    private static buildVeoApiEndpoint(): string {
        return super.buildApiEndpoint(CONFIG.VEO_MODEL, 'predictLongRunning');
    }

    private static buildFetchOperationEndpoint(): string {
        return super.buildApiEndpoint(CONFIG.VEO_MODEL, 'fetchPredictOperation');
    }

    static async generateVideo(
        imageBase64: string,
        prompt: string,
        aspectRatio: string = '16:9',
        duration: number = 8
    ): Promise<string> {
        const apiEndpoint = this.buildVeoApiEndpoint();

        // Determine if using Veo 3 model (requires generateAudio)
        const isVeo3 = CONFIG.VEO_MODEL.includes('veo-3');

        const requestBody: any = {
            instances: [
                {
                    prompt: prompt,
                    image: {
                        bytesBase64Encoded: imageBase64,
                        mimeType: 'image/png', // Default, could be detected from image
                    },
                },
            ],
            parameters: {
                aspectRatio: aspectRatio,
                durationSeconds: duration,
                sampleCount: 1,
            },
        };

        // Veo 3 models require generateAudio parameter
        if (isVeo3) {
            requestBody.parameters.generateAudio = true; // Enable audio generation
        }

        // Start the long-running operation
        interface OperationResponse {
            name?: string;
        }
        const operationResponse = await super.makeApiRequest<OperationResponse>(
            apiEndpoint,
            requestBody,
            'Veo'
        );
        const operationName = operationResponse.name;

        if (!operationName) {
            throw new Error('Failed to start video generation operation');
        }

        // Poll for completion
        const fetchEndpoint = this.buildFetchOperationEndpoint();
        const maxAttempts = 120; // 10 minutes max (5 seconds * 120)
        const pollInterval = 5000; // 5 seconds

        interface OperationStatus {
            done?: boolean;
            error?: { code?: number; message?: string; details?: unknown[] };
            response?: {
                videos?: Array<{ bytesBase64Encoded?: string; gcsUri?: string }>;
                raiMediaFilteredCount?: number;
            };
        }

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            await new Promise((resolve) => setTimeout(resolve, pollInterval));

            const operationStatus = await super.makeApiRequest<OperationStatus>(
                fetchEndpoint,
                { operationName: operationName },
                'Veo'
            );

            if (operationStatus.done) {
                // Check for errors first
                if (operationStatus.error) {
                    throw new Error(
                        `Video generation failed: ${JSON.stringify(operationStatus.error)}`
                    );
                }

                if (operationStatus.response) {
                    // Check if response has videos array
                    if (
                        operationStatus.response.videos &&
                        operationStatus.response.videos.length > 0
                    ) {
                        const video = operationStatus.response.videos[0];
                        if (video.bytesBase64Encoded) {
                            return video.bytesBase64Encoded;
                        }
                        if (video.gcsUri) {
                            throw new Error(
                                'Video stored in GCS. Please configure storageUri or download from GCS.'
                            );
                        }
                        throw new Error(
                            'Video object found but missing bytesBase64Encoded or gcsUri'
                        );
                    }

                    // Check for raiMediaFilteredCount (videos filtered by safety)
                    if (
                        operationStatus.response.raiMediaFilteredCount &&
                        operationStatus.response.raiMediaFilteredCount > 0
                    ) {
                        throw new Error(
                            `Video generation filtered by safety policies. Filtered count: ${operationStatus.response.raiMediaFilteredCount}`
                        );
                    }
                }

                throw new Error('Video generation completed but no video data found.');
            }
        }

        throw new Error('Video generation timed out');
    }
}

class MetadataService {
    private static writeMetadataToFile(metadata: Map<number, ImageMetadata>): void {
        const metadataArray = Array.from(metadata.entries()).map(([id, data]) => ({
            id,
            prompts: data.prompts,
            enhancedPrompt: data.enhancedPrompt,
            videoPrompts: data.videoPrompts || [],
        }));

        FileUtils.ensureDirectoryExists(PATHS.OUTPUTS);
        fs.writeFileSync(PATHS.METADATA, JSON.stringify(metadataArray, null, 2), 'utf-8');
    }

    static loadMetadata(): Map<number, ImageMetadata> {
        const metadata = new Map<number, ImageMetadata>();

        if (!fs.existsSync(PATHS.METADATA)) {
            return metadata;
        }

        try {
            const data = fs.readFileSync(PATHS.METADATA, 'utf-8');
            const json = JSON.parse(data);

            if (Array.isArray(json)) {
                json.forEach(
                    (item: {
                        id: number;
                        prompt?: string;
                        prompts?: string[];
                        enhancedPrompt?: string;
                        videoPrompts?: string[];
                    }) => {
                        if (item.id) {
                            // Handle both old format (single prompt) and new format (prompts array)
                            const prompts = item.prompts || (item.prompt ? [item.prompt] : []);
                            if (
                                prompts.length > 0 ||
                                (item.videoPrompts && item.videoPrompts.length > 0)
                            ) {
                                metadata.set(item.id, {
                                    prompts,
                                    enhancedPrompt: item.enhancedPrompt,
                                    videoPrompts: item.videoPrompts || [],
                                });
                            }
                        }
                    }
                );
            }
        } catch (error) {
            console.error('Error loading metadata:', error);
        }

        return metadata;
    }

    static saveMetadata(id: number, prompt: string, enhancedPrompt?: string): void {
        const metadata = this.loadMetadata();
        const existing = metadata.get(id);

        // If metadata exists, append the new prompt; otherwise create new array
        const prompts = existing?.prompts || [];
        if (!prompts.includes(prompt)) {
            prompts.push(prompt);
        }

        metadata.set(id, { prompts, enhancedPrompt });
        this.writeMetadataToFile(metadata);
    }

    static addEditPrompt(id: number, editPrompt: string): void {
        const metadata = this.loadMetadata();
        const existing = metadata.get(id);
        const prompts = existing?.prompts || [];
        const trimmedEditPrompt = editPrompt.trim();

        if (trimmedEditPrompt) {
            prompts.push(trimmedEditPrompt);
        }

        metadata.set(id, {
            prompts,
            enhancedPrompt: existing?.enhancedPrompt,
        });

        this.writeMetadataToFile(metadata);
    }

    static addVideoPrompt(id: number, videoPrompt: string): void {
        const metadata = this.loadMetadata();
        const existing = metadata.get(id);
        const videoPrompts = existing?.videoPrompts || [];
        const trimmedPrompt = videoPrompt.trim();

        if (trimmedPrompt) {
            videoPrompts.push(trimmedPrompt);
        }

        metadata.set(id, {
            prompts: existing?.prompts || [],
            enhancedPrompt: existing?.enhancedPrompt,
            videoPrompts,
        });

        this.writeMetadataToFile(metadata);
    }

    static getMetadata(id: number): ImageMetadata | undefined {
        return this.loadMetadata().get(id);
    }
}

class ImageStorageService {
    static saveImage(
        imageData: string,
        mimeType: string,
        generationId: number,
        index?: number
    ): string {
        const fileExtension = FileUtils.getFileExtension(mimeType);
        const filename =
            index !== undefined
                ? `result-${generationId}-${index}.${fileExtension}`
                : `result-${generationId}.${fileExtension}`;
        const outputPath = path.join(PATHS.OUTPUTS, filename);

        const imageBuffer = Buffer.from(imageData, 'base64');
        fs.writeFileSync(outputPath, imageBuffer);

        return `/outputs/${filename}`;
    }

    static saveImages(predictions: PredictionResponse[], generationId: number): string[] {
        return predictions.map((prediction, index) =>
            this.saveImage(
                prediction.bytesBase64Encoded,
                prediction.mimeType,
                generationId,
                predictions.length > 1 ? index : undefined
            )
        );
    }
}

class ResultService {
    static getAllResults(): ImageGenerationResult[] {
        const results: ImageGenerationResult[] = [];

        if (!fs.existsSync(PATHS.OUTPUTS)) {
            return results;
        }

        const metadata = MetadataService.loadMetadata();
        const resultFiles = fs
            .readdirSync(PATHS.OUTPUTS)
            .filter((file) => file.startsWith('result-') && !file.endsWith('.json'))
            .sort()
            .reverse();

        const resultsByGenerationId = new Map<number, { files: string[]; videos: string[] }>();

        resultFiles.forEach((file) => {
            const generationId = FileUtils.extractGenerationId(file);
            if (generationId !== null) {
                if (!resultsByGenerationId.has(generationId)) {
                    resultsByGenerationId.set(generationId, { files: [], videos: [] });
                }
                // Separate videos from images
                // Note: readdirSync already only returns files that exist, so no need for additional existsSync check
                if (file.endsWith('.mp4')) {
                    resultsByGenerationId.get(generationId)!.videos.push(`/outputs/${file}`);
                } else {
                    resultsByGenerationId.get(generationId)!.files.push(`/outputs/${file}`);
                }
            }
        });

        Array.from(resultsByGenerationId.entries())
            .sort((a, b) => b[0] - a[0])
            .forEach(([id, data]) => {
                if (data.files.length > 0) {
                    const metadataEntry = metadata.get(id);
                    results.push({
                        id,
                        prompts: metadataEntry?.prompts || [`Generated image ${id}`],
                        videoPrompts: metadataEntry?.videoPrompts || [],
                        resultImages: data.files.sort(),
                        resultVideos: data.videos.length > 0 ? data.videos.sort() : undefined,
                    });
                }
            });

        return results;
    }
}

// ============================================================================
// Request Handlers
// ============================================================================

class RequestHandlers {
    private static saveEditedImages(
        predictions: PredictionResponse[],
        originalFilename: string
    ): string[] {
        const savedImages: string[] = [];

        predictions.forEach((prediction, i) => {
            const fileExtension = FileUtils.getFileExtension(prediction.mimeType);
            const outputFilename =
                predictions.length === 1
                    ? originalFilename.replace(/\.[^.]+$/, `.${fileExtension}`)
                    : originalFilename.replace(/\.[^.]+$/, '').replace(/-(\d+)$/, '') +
                      `-${i}.${fileExtension}`;
            const outputPath = path.join(PATHS.OUTPUTS, outputFilename);

            const imageBuffer = Buffer.from(prediction.bytesBase64Encoded, 'base64');
            fs.writeFileSync(outputPath, imageBuffer);
            savedImages.push(`/outputs/${outputFilename}`);
        });

        return savedImages;
    }

    static async handleGenerateImage(req: Request, res: Response): Promise<void> {
        try {
            const body = req.body as GenerateImageRequest;
            const prompt = ValidationUtils.validatePrompt(body.prompt);
            const sampleCount = ValidationUtils.validateSampleCount(body.sampleCount);
            const aspectRatio = body.aspectRatio || CONSTRAINTS.DEFAULT_ASPECT_RATIO;
            const safetySetting = body.safetySetting || CONSTRAINTS.DEFAULT_SAFETY_SETTING;
            const personGeneration = body.personGeneration || CONSTRAINTS.DEFAULT_PERSON_GENERATION;

            const generationId = Date.now();

            const predictions = await ImageGenerationService.generateImage(
                prompt,
                sampleCount,
                aspectRatio,
                safetySetting,
                personGeneration
            );

            const savedImages = ImageStorageService.saveImages(predictions, generationId);

            const enhancedPrompt = predictions[0]?.prompt;
            MetadataService.saveMetadata(generationId, prompt, enhancedPrompt);

            const savedMetadata = MetadataService.getMetadata(generationId);

            res.json({
                success: true,
                id: generationId,
                prompts: savedMetadata?.prompts || [prompt],
                enhancedPrompt,
                resultImages: savedImages,
                count: savedImages.length,
            });
        } catch (error: unknown) {
            const errorMessage =
                error instanceof Error ? error.message : 'Failed to generate image';
            console.error('Error generating image:', error);
            res.status(500).json({
                error: errorMessage,
                details: String(error),
            });
        }
    }

    static async handleEditImage(req: Request, res: Response): Promise<void> {
        try {
            const body = req.body as EditImageRequest;
            const imagePath = ValidationUtils.validateImagePath(body.imagePath);
            const editPrompt = ValidationUtils.validatePrompt(body.editPrompt);
            const sampleCount = ValidationUtils.validateSampleCount(body.sampleCount);
            const safetySetting = body.safetySetting || CONSTRAINTS.DEFAULT_SAFETY_SETTING;
            const personGeneration = body.personGeneration || CONSTRAINTS.DEFAULT_PERSON_GENERATION;

            const filename = path.basename(imagePath);
            const generationId = FileUtils.extractGenerationId(filename);

            if (generationId === null) {
                res.status(400).json({ error: 'Invalid image path format' });
                return;
            }

            const fullImagePath = path.join(process.cwd(), imagePath.replace(/^\//, ''));

            if (!fs.existsSync(fullImagePath)) {
                res.status(404).json({ error: 'Image file not found' });
                return;
            }

            const imageBase64 = FileUtils.imageToBase64(fullImagePath);

            const predictions = await ImageGenerationService.editImage(
                imageBase64,
                editPrompt,
                sampleCount,
                safetySetting,
                personGeneration
            );

            // Delete existing video if it exists (since image was edited, old video is no longer valid)
            const videoFilename = `result-${generationId}-video.mp4`;
            const videoPath = path.join(PATHS.OUTPUTS, videoFilename);
            if (fs.existsSync(videoPath)) {
                fs.unlinkSync(videoPath);
            }

            // Save edited images (overwrite original or create numbered variants)
            const savedImages = RequestHandlers.saveEditedImages(predictions, filename);

            // Add edit prompt to metadata (preserving all previous prompts)
            MetadataService.addEditPrompt(generationId, editPrompt.trim());

            const updatedMetadata = MetadataService.getMetadata(generationId);

            res.json({
                success: true,
                id: generationId,
                prompts: updatedMetadata?.prompts || [],
                editPrompt: editPrompt.trim(),
                enhancedPrompt: predictions[0]?.prompt,
                resultImages: savedImages,
                count: savedImages.length,
            });
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to edit image';
            console.error('Error editing image:', error);
            res.status(500).json({
                error: errorMessage,
                details: String(error),
            });
        }
    }

    static async handleGenerateVideo(req: Request, res: Response): Promise<void> {
        try {
            const body = req.body as GenerateVideoRequest;
            const imagePath = ValidationUtils.validateImagePath(body.imagePath);
            const prompt = ValidationUtils.validatePrompt(body.prompt);
            const aspectRatio = body.aspectRatio || '16:9';
            const duration = body.duration || 8; // Default 8 seconds for Veo 3 models image-to-video (supported: 4, 6, or 8)

            const filename = path.basename(imagePath);
            const generationId = FileUtils.extractGenerationId(filename);

            if (generationId === null) {
                res.status(400).json({ error: 'Invalid image path format' });
                return;
            }

            const fullImagePath = path.join(process.cwd(), imagePath.replace(/^\//, ''));

            if (!fs.existsSync(fullImagePath)) {
                res.status(404).json({ error: 'Image file not found' });
                return;
            }

            const imageBase64 = FileUtils.imageToBase64(fullImagePath);

            // Generate video (this is a long-running operation)
            const videoBase64 = await VideoGenerationService.generateVideo(
                imageBase64,
                prompt,
                aspectRatio,
                duration
            );

            // Save video (overwrite if exists)
            const videoFilename = `result-${generationId}-video.mp4`;
            const videoPath = path.join(PATHS.OUTPUTS, videoFilename);

            // Delete existing video if it exists to ensure clean overwrite
            if (fs.existsSync(videoPath)) {
                fs.unlinkSync(videoPath);
            }

            const videoBuffer = Buffer.from(videoBase64, 'base64');
            fs.writeFileSync(videoPath, videoBuffer);

            // Add video prompt to metadata
            MetadataService.addVideoPrompt(generationId, prompt);

            res.json({
                success: true,
                id: generationId,
                videoUrl: `/outputs/${videoFilename}`,
                prompt: prompt.trim(),
            });
        } catch (error: unknown) {
            const errorMessage =
                error instanceof Error ? error.message : 'Failed to generate video';
            console.error('Error generating video:', error);
            res.status(500).json({
                error: errorMessage,
                details: String(error),
            });
        }
    }

    static async handleGetResults(_req: Request, res: Response): Promise<void> {
        try {
            const results = ResultService.getAllResults();
            res.json({ results });
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get results';
            console.error('Error getting results:', error);
            res.status(500).json({
                error: errorMessage,
                details: String(error),
            });
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

function initializeApp(): void {
    FileUtils.ensureDirectoryExists(PATHS.OUTPUTS);

    if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
        if (!fs.existsSync(PATHS.SERVICE_ACCOUNT_KEY)) {
            throw new Error(`Service account key file not found at: ${PATHS.SERVICE_ACCOUNT_KEY}`);
        }
        process.env.GOOGLE_APPLICATION_CREDENTIALS = PATHS.SERVICE_ACCOUNT_KEY;
    }
}

// ============================================================================
// Express App Setup
// ============================================================================

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Serve outputs with no-cache headers to prevent browser caching of edited images
app.use('/outputs', (req, res, next) => {
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    express.static(PATHS.OUTPUTS)(req, res, next);
});

// ============================================================================
// API Routes
// ============================================================================

app.post('/api/generate-image', RequestHandlers.handleGenerateImage);
app.post('/api/edit-image', RequestHandlers.handleEditImage);
app.post('/api/generate-video', RequestHandlers.handleGenerateVideo);
app.get('/api/results', RequestHandlers.handleGetResults);

// ============================================================================
// Server Startup
// ============================================================================

initializeApp();

app.listen(CONFIG.PORT, () => {
    console.log(`🚀 Server running on http://localhost:${CONFIG.PORT}`);
    console.log(`📁 Outputs directory: ${PATHS.OUTPUTS}`);
    console.log(`🎨 Using Imagen model: ${CONFIG.IMAGEN_MODEL}`);
    console.log(`✏️  Using Imagen Edit model: ${CONFIG.IMAGEN_EDIT_MODEL}`);
});
