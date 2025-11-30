export interface SegmentationResult {
  overlayData: string; // Base64 encoded image or data URL
  statistics: {
    [key: string]: number | { pixels: number; percentage: number };
  };
}
