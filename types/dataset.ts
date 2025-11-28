export interface DatasetEntry {
  id: string;
  filename: string;
  courseName: string;
  timestamp: string;
  location: {
    name: string;
    lat: number;
    lng: number;
    zoomLevel: number;
  };
  captureBox: {
    width: number;
    height: number;
    bounds: {
      north: number;
      south: number;
      east: number;
      west: number;
    };
  };
  images: {
    satellite?: string; // Base64 data URL
    mask?: string; // Base64 data URL
    groundTruth?: string; // Base64 data URL
  };
  segmentation?: {
    model: string;
    version: string;
    hasPrediction: boolean;
    hasGroundTruth: boolean;
    classDistribution: {
      [key: string]: number;
    };
  };
  userNotes?: string;
}

export interface SaveOptions {
  saveImage: boolean;
  saveMask: boolean;
  saveGroundTruth: boolean;
}

export interface ExportFormat {
  type: 'zip' | 'folder' | 'cloud';
  structure: 'training-ready' | 'tensorflow' | 'coco';
  include: {
    images: boolean;
    masks: boolean;
    metadata: boolean;
    readme: boolean;
  };
}
