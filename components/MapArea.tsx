'use client';

import { useEffect, useRef, useState } from 'react';
import { Loader } from '@googlemaps/js-api-loader';
import { SegmentationResult } from '../app/page';
import { toast } from 'sonner';
import { Lightbulb } from 'lucide-react';
import html2canvas from 'html2canvas';

interface MapAreaProps {
  center: { lat: number; lng: number };
  segmentationResult: SegmentationResult | null;
  isSegmenting: boolean;
  overlayOpacity: number;
  showBox: boolean;
  triggerCapture?: boolean;
  onSegmentationComplete: (result: SegmentationResult) => void;
  onImageCapture?: (imageData: string) => void;
  onCaptureComplete?: () => void;
}

const CAPTURE_BOX_WIDTH = 1664;
const CAPTURE_BOX_HEIGHT = 1024;

// Segmentation API URL - Hugging Face Spaces
const SEGMENTATION_API_URL = process.env.NEXT_PUBLIC_SEGMENTATION_API_URL || 'https://elo0oo0-golf-segmentation-api.hf.space';

// Real segmentation function - calls FastAPI backend
async function performSegmentation(imageBlob: Blob): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'map_capture.jpg');

  const response = await fetch(`${SEGMENTATION_API_URL}/segment`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Segmentation failed: ${response.statusText}`);
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error('Segmentation API returned failure');
  }

  return {
    overlayData: data.overlayData,
    statistics: data.statistics,
  };
}

export function MapArea({
  center,
  segmentationResult,
  isSegmenting,
  overlayOpacity,
  showBox,
  triggerCapture,
  onSegmentationComplete,
  onImageCapture,
  onCaptureComplete,
}: MapAreaProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const captureBoxRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<google.maps.Map | null>(null);
  const overlayRef = useRef<google.maps.GroundOverlay | null>(null);
  const overlayBoundsRef = useRef<google.maps.LatLngBounds | null>(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  // Initialize Google Maps
  useEffect(() => {
    // Get API key - handle different environments
    const apiKey = typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
      ? process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
      : 'YOUR_GOOGLE_MAPS_API_KEY';

    // Initialize the Google Maps loader
    const loader = new Loader({
      apiKey,
      version: 'weekly',
      libraries: ['maps', 'marker']
    });

    // Load and initialize the maps library
    loader
      .load()
      .then(async () => {
        if (mapRef.current && !mapInstanceRef.current) {
          const { Map } = await google.maps.importLibrary('maps') as google.maps.MapsLibrary;

          const map = new Map(mapRef.current, {
            center,
            zoom: 17,
            mapId: 'satellite',
            mapTypeId: 'satellite',
            disableDefaultUI: false,
            zoomControl: true,
            mapTypeControl: true,
            streetViewControl: false,
            fullscreenControl: true,
          });

          mapInstanceRef.current = map;
          setIsMapLoaded(true);
        }
      })
      .catch(() => {
        toast.error('Failed to load Google Maps. Using mock map.');
        setIsMapLoaded(true);
      });
  }, []);

  // Update map center when location changes
  useEffect(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setCenter(center);
      mapInstanceRef.current.setZoom(17);
    }
  }, [center]);

  // Handle dataset capture (just capture image, no segmentation)
  useEffect(() => {
    if (triggerCapture) {
      const captureOnly = async () => {
        try {
          if (!mapRef.current || !captureBoxRef.current) {
            throw new Error('Map or capture box reference not available');
          }

          // Hide Google Maps controls during capture
          if (mapInstanceRef.current) {
            mapInstanceRef.current.setOptions({
              disableDefaultUI: true,
              zoomControl: false,
              mapTypeControl: false,
              streetViewControl: false,
              fullscreenControl: false,
            });
          }

          // Wait for controls to hide
          await new Promise(resolve => setTimeout(resolve, 100));

          // Get the actual positions of the map and capture box
          const mapRect = mapRef.current.getBoundingClientRect();
          const boxRect = captureBoxRef.current.getBoundingClientRect();

          // Capture the entire map div
          const fullCanvas = await html2canvas(mapRef.current, {
            useCORS: true,
            allowTaint: true,
            logging: false,
            scale: 1,
          });

          // Re-enable Google Maps controls
          if (mapInstanceRef.current) {
            mapInstanceRef.current.setOptions({
              disableDefaultUI: false,
              zoomControl: true,
              mapTypeControl: true,
              fullscreenControl: true,
            });
          }

          // Calculate crop coordinates relative to the map
          const cropX = boxRect.left - mapRect.left;
          const cropY = boxRect.top - mapRect.top;
          const cropWidth = boxRect.width;
          const cropHeight = boxRect.height;

          // Create a new canvas with the capture box dimensions
          const croppedCanvas = document.createElement('canvas');
          croppedCanvas.width = cropWidth;
          croppedCanvas.height = cropHeight;
          const ctx = croppedCanvas.getContext('2d')!;

          // Draw the cropped portion from the full canvas
          ctx.drawImage(
            fullCanvas,
            cropX, cropY, cropWidth, cropHeight,
            0, 0, cropWidth, cropHeight
          );

          // Save the image data
          const imageData = croppedCanvas.toDataURL('image/jpeg', 0.9);
          if (onImageCapture) {
            onImageCapture(imageData);
          }

          // Signal capture is complete
          if (onCaptureComplete) {
            onCaptureComplete();
          }
        } catch (error) {
          console.error('Capture error:', error);
          toast.error('Failed to capture image');
          if (onCaptureComplete) {
            onCaptureComplete();
          }
        }
      };

      captureOnly();
    }
  }, [triggerCapture, onImageCapture, onCaptureComplete]);

  // Handle segmentation
  useEffect(() => {
    if (isSegmenting) {
      // Capture the map image and perform segmentation
      const captureAndSegment = async () => {
        try {
          if (!mapRef.current || !captureBoxRef.current) {
            throw new Error('Map or capture box reference not available');
          }

          // Hide Google Maps controls during capture
          if (mapInstanceRef.current) {
            mapInstanceRef.current.setOptions({
              disableDefaultUI: true,
              zoomControl: false,
              mapTypeControl: false,
              streetViewControl: false,
              fullscreenControl: false,
            });
          }

          // Wait for controls to hide
          await new Promise(resolve => setTimeout(resolve, 100));

          // Get the actual positions of the map and capture box
          const mapRect = mapRef.current.getBoundingClientRect();
          const boxRect = captureBoxRef.current.getBoundingClientRect();

          // Calculate the lat/lng bounds of the capture box
          if (mapInstanceRef.current) {
            const projection = mapInstanceRef.current.getProjection();
            const bounds = mapInstanceRef.current.getBounds();

            if (projection && bounds) {
              // Get the map's top-left corner in lat/lng
              const ne = bounds.getNorthEast();
              const sw = bounds.getSouthWest();

              // Calculate the pixel-to-latlng ratio
              const latPerPixel = (ne.lat() - sw.lat()) / mapRect.height;
              const lngPerPixel = (ne.lng() - sw.lng()) / mapRect.width;

              // Calculate bounds based on box position relative to map
              const boxLeftOffset = boxRect.left - mapRect.left;
              const boxTopOffset = boxRect.top - mapRect.top;
              const boxRightOffset = boxLeftOffset + boxRect.width;
              const boxBottomOffset = boxTopOffset + boxRect.height;

              // Convert pixel offsets to lat/lng
              const west = sw.lng() + (boxLeftOffset * lngPerPixel);
              const east = sw.lng() + (boxRightOffset * lngPerPixel);
              const north = ne.lat() - (boxTopOffset * latPerPixel);
              const south = ne.lat() - (boxBottomOffset * latPerPixel);

              // Store the bounds for overlay rendering
              overlayBoundsRef.current = new google.maps.LatLngBounds(
                new google.maps.LatLng(south, west),
                new google.maps.LatLng(north, east)
              );
            }
          }

          // Capture the entire map div
          const fullCanvas = await html2canvas(mapRef.current, {
            useCORS: true,
            allowTaint: true,
            logging: false,
            scale: 1,
          });

          // Re-enable Google Maps controls
          if (mapInstanceRef.current) {
            mapInstanceRef.current.setOptions({
              disableDefaultUI: false,
              zoomControl: true,
              mapTypeControl: true,
              fullscreenControl: true,
            });
          }

          // Calculate crop coordinates relative to the map
          // The box position is absolute to viewport, so subtract map's position
          const cropX = boxRect.left - mapRect.left;
          const cropY = boxRect.top - mapRect.top;
          const cropWidth = boxRect.width;
          const cropHeight = boxRect.height;

          // Create a new canvas with the capture box dimensions
          const croppedCanvas = document.createElement('canvas');
          croppedCanvas.width = cropWidth;
          croppedCanvas.height = cropHeight;
          const ctx = croppedCanvas.getContext('2d')!;

          // Draw the cropped portion from the full canvas
          ctx.drawImage(
            fullCanvas,
            cropX, cropY, cropWidth, cropHeight,  // Source rectangle from full canvas
            0, 0, cropWidth, cropHeight            // Destination (entire cropped canvas)
          );

          // Save the image data for display
          const imageData = croppedCanvas.toDataURL('image/jpeg', 0.9);
          if (onImageCapture) {
            onImageCapture(imageData);
          }

          // Convert canvas to Blob for API
          const blob = await new Promise<Blob>((resolve) => {
            croppedCanvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.9);
          });

          // Call the real segmentation API
          const result = await performSegmentation(blob);
          onSegmentationComplete(result);
          toast.success('Segmentation complete!');
        } catch (error) {
          console.error('Segmentation error:', error);
          toast.error('Error: Failed to process image. Make sure the API is running.');
          // Reset segmenting state on error
          onSegmentationComplete({ overlayData: '', statistics: {} });
        }
      };

      captureAndSegment();
    }
  }, [isSegmenting, onSegmentationComplete, onImageCapture]);

  // Update overlay when segmentation result or opacity changes
  useEffect(() => {
    console.log('Overlay effect triggered:', {
      hasMap: !!mapInstanceRef.current,
      hasResult: !!segmentationResult,
      hasBounds: !!overlayBoundsRef.current,
      overlayDataLength: segmentationResult?.overlayData?.length || 0,
    });

    if (mapInstanceRef.current && segmentationResult && segmentationResult.overlayData) {
      // Remove existing overlay
      if (overlayRef.current) {
        overlayRef.current.setMap(null);
      }

      // If no bounds were captured, create default bounds around current center
      let bounds = overlayBoundsRef.current;
      if (!bounds) {
        const center = mapInstanceRef.current.getCenter();
        if (center) {
          // Create bounds based on approximate capture area
          const lat = center.lat();
          const lng = center.lng();
          const latOffset = 0.01; // Approximate degrees for overlay size
          const lngOffset = 0.015;
          bounds = new google.maps.LatLngBounds(
            new google.maps.LatLng(lat - latOffset, lng - lngOffset),
            new google.maps.LatLng(lat + latOffset, lng + lngOffset)
          );
        }
      }

      if (bounds) {
        // Use the exact bounds that were calculated during capture
        const overlay = new google.maps.GroundOverlay(
          segmentationResult.overlayData,
          bounds,
          {
            opacity: overlayOpacity / 100,
          }
        );

        overlay.setMap(mapInstanceRef.current);
        overlayRef.current = overlay;
        console.log('Overlay created successfully');
      }
    }
  }, [segmentationResult, overlayOpacity]);

  return (
    <div className="flex-1 relative">
      {/* Map Container */}
      <div ref={mapRef} className="w-full h-full bg-slate-900" />

      {/* Fallback when Maps API not loaded */}
      {!isMapLoaded && (
        <div className="absolute inset-0 bg-slate-900 flex items-center justify-center">
          <div className="text-center">
            <div className="size-8 border-2 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-slate-300">Loading map...</p>
            <p className="text-slate-500 text-sm mt-2">Add your Google Maps API key to use real maps</p>
          </div>
        </div>
      )}

      {/* Capture Box Overlay */}
      {showBox && !segmentationResult && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            ref={captureBoxRef}
            className="relative border-[3px] border-dashed border-yellow-400 bg-black/20"
            style={{
              width: `${CAPTURE_BOX_WIDTH}px`,
              height: `${CAPTURE_BOX_HEIGHT}px`,
              maxWidth: '90vw',
              maxHeight: '80vh',
            }}
          >
            {/* Corner markers */}
            <div className="absolute -top-2 -left-2 size-4 bg-yellow-400 rounded-full" />
            <div className="absolute -top-2 -right-2 size-4 bg-yellow-400 rounded-full" />
            <div className="absolute -bottom-2 -left-2 size-4 bg-yellow-400 rounded-full" />
            <div className="absolute -bottom-2 -right-2 size-4 bg-yellow-400 rounded-full" />

            {/* Label */}
            <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
              Capture Area: {CAPTURE_BOX_WIDTH} × {CAPTURE_BOX_HEIGHT}
            </div>

            {/* Dimensions indicator */}
            <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 bg-slate-800 text-white px-3 py-1 rounded text-sm whitespace-nowrap">
              {CAPTURE_BOX_WIDTH} × {CAPTURE_BOX_HEIGHT} px
            </div>
          </div>
        </div>
      )}

      {/* Helper text */}
      {!segmentationResult && !isSegmenting && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 bg-slate-800/90 text-slate-200 px-4 py-3 rounded-lg flex items-center gap-2 shadow-lg pointer-events-none max-w-md text-center">
          <Lightbulb className="size-5 text-yellow-400 flex-shrink-0" />
          <span className="text-sm">
            Position a golf course within the box and click "Segment Course"
          </span>
        </div>
      )}

      {/* Loading overlay */}
      {isSegmenting && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
          <div className="bg-slate-800 rounded-lg p-6 text-center">
            <div className="size-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-white mb-2">Analyzing...</p>
            <div className="w-64 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
            <p className="text-slate-400 text-sm mt-2">Processing image...</p>
          </div>
        </div>
      )}
    </div>
  );
}