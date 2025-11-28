'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Undo2, Redo2, Eraser, Save, X, Eye, EyeOff, Sparkles, MousePointer2, Paintbrush, MessageSquare, Send, Bot, ChevronRight, ChevronLeft, Check, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

interface AnnotationCanvasProps {
  imageData: string;
  initialAnnotation?: string;
  onSave: (annotationData: string) => void;
  onClose: () => void;
}

const ANNOTATION_CLASSES = [
  { name: 'Background', color: '#1a1a1a', key: '1' },
  { name: 'Fairway', color: '#2d5016', key: '2' },
  { name: 'Green', color: '#4ade80', key: '3' },
  { name: 'Tee', color: '#ef4444', key: '4' },
  { name: 'Bunker', color: '#fbbf24', key: '5' },
  { name: 'Water', color: '#3b82f6', key: '6' },
];

interface HistoryState {
  imageData: string;
}

interface AiMessage {
  role: 'assistant' | 'user';
  content: string;
  timestamp: Date;
  features?: { class: string; centerX: number; centerY: number; radius: number }[];
  colorProfiles?: { [key: string]: any };
}

interface AiAnalysis {
  features: { class: string; centerX: number; centerY: number; radius: number }[];
  colorProfiles: { [key: string]: any };
}

export function AnnotationCanvas({ imageData, initialAnnotation, onSave, onClose }: AnnotationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectedClass, setSelectedClass] = useState(ANNOTATION_CLASSES[1]); // Default to Fairway
  const [brushSize, setBrushSize] = useState(20);
  const [isEraser, setIsEraser] = useState(false);
  const [opacity, setOpacity] = useState(60);
  const [showOverlay, setShowOverlay] = useState(true);
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [zoom, setZoom] = useState(100);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [isSamProcessing, setIsSamProcessing] = useState(false);

  // AI Chat Panel State
  const [showAiPanel, setShowAiPanel] = useState(true);
  const [aiMessages, setAiMessages] = useState<AiMessage[]>([]);
  const [userFeedback, setUserFeedback] = useState('');
  const [currentAnalysis, setCurrentAnalysis] = useState<AiAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!canvas || !overlayCanvas) return;

    const ctx = canvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');
    if (!ctx || !overlayCtx) return;

    // Load the satellite image
    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;
      overlayCanvas.width = img.width;
      overlayCanvas.height = img.height;

      // Draw background image on main canvas
      ctx.drawImage(img, 0, 0);

      // Initialize overlay with transparent background
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Load initial annotation if provided
      if (initialAnnotation) {
        const annotationImg = new Image();
        annotationImg.onload = () => {
          overlayCtx.drawImage(annotationImg, 0, 0);
          saveToHistory();
        };
        annotationImg.src = initialAnnotation;
      } else {
        // Save initial state
        saveToHistory();
      }
    };
    img.src = imageData;
  }, [imageData, initialAnnotation]);

  // Save state to history
  const saveToHistory = useCallback(() => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const newState: HistoryState = {
      imageData: overlayCanvas.toDataURL(),
    };

    // Remove any states after current index (when undoing then making new changes)
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newState);

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift();
    } else {
      setHistoryIndex(historyIndex + 1);
    }

    setHistory(newHistory);
  }, [history, historyIndex]);

  // Undo
  const handleUndo = useCallback(() => {
    if (historyIndex <= 0) return;

    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    const newIndex = historyIndex - 1;
    const state = history[newIndex];

    const img = new Image();
    img.onload = () => {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.drawImage(img, 0, 0);
    };
    img.src = state.imageData;

    setHistoryIndex(newIndex);
  }, [history, historyIndex]);

  // Redo
  const handleRedo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    const newIndex = historyIndex + 1;
    const state = history[newIndex];

    const img = new Image();
    img.onload = () => {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.drawImage(img, 0, 0);
    };
    img.src = state.imageData;

    setHistoryIndex(newIndex);
  }, [history, historyIndex]);

  // Clear all annotations
  const handleClear = useCallback(() => {
    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    saveToHistory();
    toast.success('Annotations cleared');
  }, [saveToHistory]);

  // Drawing functions
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    setIsDrawing(true);
    const rect = overlayCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (overlayCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (overlayCanvas.height / rect.height);
    
    draw(x, y);
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveToHistory();
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Update cursor position for brush preview
    setCursorPos({ x: e.clientX, y: e.clientY });

    // Handle panning
    if (isPanning && e.buttons === 4) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      setPanOffset({ x: panOffset.x + dx, y: panOffset.y + dy });
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    if (!isDrawing) return;

    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const rect = overlayCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (overlayCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (overlayCanvas.height / rect.height);

    draw(x, y);
  };

  // Mouse wheel zoom
  const handleWheel = useCallback((e: WheelEvent) => {
    if (!containerRef.current) return;

    e.preventDefault();

    const delta = e.deltaY > 0 ? -10 : 10;
    const newZoom = Math.min(400, Math.max(50, zoom + delta));

    if (newZoom !== zoom) {
      // Calculate zoom toward mouse position
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - rect.width / 2;
      const mouseY = e.clientY - rect.top - rect.height / 2;

      const zoomFactor = newZoom / zoom;
      const newPanX = panOffset.x - mouseX * (zoomFactor - 1);
      const newPanY = panOffset.y - mouseY * (zoomFactor - 1);

      setZoom(newZoom);
      setPanOffset({ x: newPanX, y: newPanY });
    }
  }, [zoom, panOffset]);

  // Handle middle mouse button pan
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 1) { // Middle mouse button
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }
    startDrawing(e);
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 1) {
      setIsPanning(false);
      return;
    }
    stopDrawing();
  };

  const draw = (x: number, y: number) => {
    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCtx) return;

    overlayCtx.beginPath();
    overlayCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);

    if (isEraser) {
      overlayCtx.globalCompositeOperation = 'destination-out';
      overlayCtx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      overlayCtx.globalCompositeOperation = 'source-over';
      overlayCtx.fillStyle = selectedClass.color;
    }

    overlayCtx.fill();
  };

  // Simple Segment Everything - just runs SAM and applies ALL masks as selected class
  const handleSegmentEverything = async () => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    toast.info(`🎯 SAM 2 segmenting everything as ${selectedClass.name}...`);

    try {
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: 'all' }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions.');
        return;
      }

      toast.info(`🔍 Applying ${individualMasks.length} regions as ${selectedClass.name}...`);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing painted pixels to respect
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();
      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0) {
          paintedPixels.add(i / 4);
        }
      }

      const classColor = hexToRgb(selectedClass.color);
      let appliedCount = 0;

      // Apply ALL masks as the selected class
      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
          const pixels = currentData.data;

          let applied = false;
          for (let j = 0; j < maskData.data.length; j += 4) {
            const pixelIndex = j / 4;
            if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
              pixels[j] = classColor.r;
              pixels[j + 1] = classColor.g;
              pixels[j + 2] = classColor.b;
              pixels[j + 3] = 255;
              paintedPixels.add(pixelIndex);
              applied = true;
            }
          }

          if (applied) {
            ctx.putImageData(currentData, 0, 0);
            appliedCount++;
          }
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      saveToHistory();
      toast.success(`✨ Applied ${appliedCount} regions as ${selectedClass.name}. Use eraser to refine.`);

    } catch (error: any) {
      console.error('Segment everything error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [aiMessages]);

  // AI Analysis - Get suggestions from GPT-4 without applying
  const handleAiAnalyze = async (feedback?: string) => {
    setIsAnalyzing(true);

    // Add user feedback message if provided
    if (feedback) {
      setAiMessages(prev => [...prev, {
        role: 'user',
        content: feedback,
        timestamp: new Date(),
      }]);
      setUserFeedback('');
    }

    try {
      const gptResponse = await fetch('/api/gpt-classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageData,
          feedback: feedback || undefined,
        }),
      });

      if (!gptResponse.ok) {
        throw new Error('Failed to analyze image');
      }

      const data = await gptResponse.json();
      const features = data.features || [];
      const colorProfiles = data.colorProfiles || {};

      setCurrentAnalysis({ features, colorProfiles });

      // Build analysis message
      const featureCounts: { [key: string]: number } = {};
      features.forEach((f: any) => {
        featureCounts[f.class] = (featureCounts[f.class] || 0) + 1;
      });

      const featureSummary = Object.entries(featureCounts)
        .map(([cls, count]) => `${count} ${cls}${count > 1 ? 's' : ''}`)
        .join(', ') || 'No features detected';

      // Build detailed color profile description
      const colorClasses = Object.keys(colorProfiles);
      const colorDetails = colorClasses.map(cls => {
        const cp = colorProfiles[cls];
        return `**${cls}:**\n  R: ${cp.minR}-${cp.maxR} | G: ${cp.minG}-${cp.maxG} | B: ${cp.minB}-${cp.maxB}\n  Brightness: ${cp.minBrightness}-${cp.maxBrightness}`;
      }).join('\n\n');

      const analysisMessage = `**Analysis Complete**\n\n**Feature Locations:** ${featureSummary}\n\n**Color Profiles (RGB ranges for this image):**\n\n${colorDetails || 'No color profiles detected'}\n\nClick "Apply" to segment with these colors, or provide feedback to refine.`;

      setAiMessages(prev => [...prev, {
        role: 'assistant',
        content: analysisMessage,
        timestamp: new Date(),
        features,
        colorProfiles,
      }]);

    } catch (error: any) {
      setAiMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Analysis failed: ${error.message}`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Apply the current AI analysis with SAM
  const handleApplyAnalysis = async () => {
    if (!currentAnalysis) {
      toast.warning('Run analysis first');
      return;
    }

    // Use the stored analysis to run segmentation
    setIsSamProcessing(true);
    toast.info('🎯 Applying AI analysis with SAM 2...');

    try {
      const overlayCanvas = overlayCanvasRef.current;
      const baseCanvas = canvasRef.current;
      if (!overlayCanvas || !baseCanvas) return;

      const { features: gptFeatures, colorProfiles: gptColorProfiles } = currentAnalysis;

      // Get SAM masks
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: 'all' }),
      });

      if (!response.ok) {
        throw new Error('SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions');
        return;
      }

      const baseCtx = baseCanvas.getContext('2d')!;
      const originalImageData = baseCtx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);
      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing painted pixels
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();
      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0) {
          paintedPixels.add(i / 4);
        }
      }

      // Helper functions (using the existing ones from the component)
      const findGptMatch = (maskCenterX: number, maskCenterY: number): string | null => {
        if (gptFeatures.length === 0) return null;
        const normX = (maskCenterX / overlayCanvas.width) * 100;
        const normY = (maskCenterY / overlayCanvas.height) * 100;
        for (const feature of gptFeatures) {
          const distance = Math.sqrt(
            Math.pow(normX - feature.centerX, 2) + Math.pow(normY - feature.centerY, 2)
          );
          if (distance < feature.radius + 10) return feature.class;
        }
        return null;
      };

      const getMaskCenter = (maskData: ImageData): { x: number; y: number } => {
        let sumX = 0, sumY = 0, count = 0;
        const width = maskData.width;
        for (let i = 0; i < maskData.data.length; i += 4) {
          if (maskData.data[i] > 128) {
            const pixelIndex = i / 4;
            sumX += pixelIndex % width;
            sumY += Math.floor(pixelIndex / width);
            count++;
          }
        }
        return count > 0 ? { x: sumX / count, y: sumY / count } : { x: 0, y: 0 };
      };

      const classifiedMasks: { maskUrl: string; className: string; priority: number }[] = [];

      for (const maskUrl of individualMasks) {
        if (!maskUrl) continue;
        try {
          const maskImg = await loadImage(maskUrl);
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const colors = analyzeColorsInMask(originalImageData, maskData);
          const maskCenter = getMaskCenter(maskData);
          const gptClass = findGptMatch(maskCenter.x, maskCenter.y);

          let className: string;
          if (gptClass && isColorCompatible(colors, gptClass)) {
            className = gptClass;
          } else {
            className = classifyWithCustomProfiles(colors, gptColorProfiles);
          }

          classifiedMasks.push({ maskUrl, className, priority: getClassPriority(className) });
        } catch (e) {
          console.warn('Failed to process mask:', e);
        }
      }

      classifiedMasks.sort((a, b) => a.priority - b.priority);

      const counts: { [key: string]: number } = {};
      for (const { maskUrl, className } of classifiedMasks) {
        if (className === 'Background') continue;
        const classInfo = ANNOTATION_CLASSES.find(c => c.name === className);
        if (!classInfo) continue;

        const maskImg = await loadImage(maskUrl);
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = overlayCanvas.width;
        tempCanvas.height = overlayCanvas.height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

        const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
        const pixels = currentData.data;
        const classColor = hexToRgb(classInfo.color);

        let applied = false;
        for (let j = 0; j < maskData.data.length; j += 4) {
          const pixelIndex = j / 4;
          if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
            pixels[j] = classColor.r;
            pixels[j + 1] = classColor.g;
            pixels[j + 2] = classColor.b;
            pixels[j + 3] = 255;
            paintedPixels.add(pixelIndex);
            applied = true;
          }
        }

        if (applied) {
          ctx.putImageData(currentData, 0, 0);
          counts[className] = (counts[className] || 0) + 1;
        }
      }

      saveToHistory();
      const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ');

      setAiMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ **Applied!** Found: ${summary || 'no features'}`,
        timestamp: new Date(),
      }]);

      toast.success(`Applied: ${summary || 'no features'}`);

    } catch (error: any) {
      toast.error(`Failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Smart Segment All - uses GPT-4 Vision + SAM for accurate classification
  const handleSmartSegmentAll = async () => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    setShowAiPanel(true); // Open the AI panel to show reasoning
    toast.info('🧠 GPT-4 Vision analyzing golf course layout...');

    // Add initial message
    setAiMessages(prev => [...prev, {
      role: 'assistant',
      content: '🔍 **Analyzing image...**\n\nScanning for golf course features...',
      timestamp: new Date(),
    }]);

    try {
      // Step 1: Get GPT-4 Vision to identify feature locations AND color profiles for this specific image
      let gptFeatures: { class: string; centerX: number; centerY: number; radius: number }[] = [];
      let gptColorProfiles: { [key: string]: {
        minR: number; maxR: number;
        minG: number; maxG: number;
        minB: number; maxB: number;
        minBrightness: number; maxBrightness: number;
      }} = {};

      try {
        const gptResponse = await fetch('/api/gpt-classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imageData }),
        });

        if (gptResponse.ok) {
          const gptData = await gptResponse.json();
          gptFeatures = gptData.features || [];
          gptColorProfiles = gptData.colorProfiles || {};

          // Store analysis for potential reuse
          setCurrentAnalysis({ features: gptFeatures, colorProfiles: gptColorProfiles });

          // Build analysis message
          const featureCounts: { [key: string]: number } = {};
          gptFeatures.forEach((f) => {
            featureCounts[f.class] = (featureCounts[f.class] || 0) + 1;
          });

          const featureSummary = Object.entries(featureCounts)
            .map(([cls, count]) => `${count} ${cls}${count > 1 ? 's' : ''}`)
            .join(', ') || 'No specific features';

          const colorClasses = Object.keys(gptColorProfiles);

          // Build detailed color profile description
          const colorDetails = colorClasses.map(cls => {
            const cp = gptColorProfiles[cls];
            return `**${cls}:**\n  R: ${cp.minR}-${cp.maxR} | G: ${cp.minG}-${cp.maxG} | B: ${cp.minB}-${cp.maxB}\n  Brightness: ${cp.minBrightness}-${cp.maxBrightness}`;
          }).join('\n\n');

          // Add GPT analysis message to chat
          setAiMessages(prev => [...prev, {
            role: 'assistant',
            content: `**GPT-4 Analysis Complete**\n\n**Feature Locations:** ${featureSummary}\n\n**Color Profiles (RGB ranges for this image):**\n\n${colorDetails || 'No color profiles detected'}\n\n⏳ Running SAM 2 segmentation with these colors...`,
            timestamp: new Date(),
            features: gptFeatures,
            colorProfiles: gptColorProfiles,
          }]);

          toast.info(`🎯 GPT-4: ${gptFeatures.length} features, ${colorClasses.length} color profiles. Running SAM 2...`);
          console.log('GPT-4 color profiles for this image:', gptColorProfiles);
        }
      } catch (e) {
        console.warn('GPT-4 Vision failed, falling back to global colors:', e);
        setAiMessages(prev => [...prev, {
          role: 'assistant',
          content: '⚠️ GPT-4 analysis failed, using default color detection...',
          timestamp: new Date(),
        }]);
      }

      // Step 2: Get SAM masks
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: 'all' }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions. Try a different image.');
        return;
      }

      toast.info(`🔍 Analyzing ${individualMasks.length} regions...`);

      const baseCtx = baseCanvas.getContext('2d')!;
      const originalImageData = baseCtx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing annotation data to preserve already-painted areas
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();

      // Mark already painted pixels (check alpha OR any RGB value)
      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0 || existingData.data[i] > 0 || existingData.data[i + 1] > 0 || existingData.data[i + 2] > 0) {
          paintedPixels.add(i / 4);
        }
      }

      // Helper: Check if mask center is near a GPT-identified feature
      const findGptMatch = (maskCenterX: number, maskCenterY: number): string | null => {
        if (gptFeatures.length === 0) return null;

        // Normalize coordinates to 0-100 range
        const normX = (maskCenterX / overlayCanvas.width) * 100;
        const normY = (maskCenterY / overlayCanvas.height) * 100;

        for (const feature of gptFeatures) {
          const distance = Math.sqrt(
            Math.pow(normX - feature.centerX, 2) + Math.pow(normY - feature.centerY, 2)
          );
          // If mask center is within the feature's radius, it's a match
          if (distance < feature.radius + 10) {
            return feature.class;
          }
        }
        return null;
      };

      // Helper: Calculate mask center
      const getMaskCenter = (maskData: ImageData): { x: number; y: number } => {
        let sumX = 0, sumY = 0, count = 0;
        const width = maskData.width;
        for (let i = 0; i < maskData.data.length; i += 4) {
          if (maskData.data[i] > 128) {
            const pixelIndex = i / 4;
            const x = pixelIndex % width;
            const y = Math.floor(pixelIndex / width);
            sumX += x;
            sumY += y;
            count++;
          }
        }
        return count > 0 ? { x: sumX / count, y: sumY / count } : { x: 0, y: 0 };
      };

      // Classify each mask using GPT context + color
      const classifiedMasks: { maskUrl: string; className: string; priority: number }[] = [];

      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const colors = analyzeColorsInMask(originalImageData, maskData);

          // First try GPT-4's spatial identification
          const maskCenter = getMaskCenter(maskData);
          const gptClass = findGptMatch(maskCenter.x, maskCenter.y);

          // Use GPT class if available and the color is compatible, otherwise fall back to color
          let className: string;
          if (gptClass && isColorCompatible(colors, gptClass)) {
            className = gptClass;
          } else {
            // Use GPT's custom color profiles if available, otherwise use global classification
            className = classifyWithCustomProfiles(colors, gptColorProfiles);
          }

          const priority = getClassPriority(className);
          classifiedMasks.push({ maskUrl, className, priority });
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      // Sort by priority (Water=1 first, Fairway=5 last)
      classifiedMasks.sort((a, b) => a.priority - b.priority);

      const counts: { [key: string]: number } = {};

      // Apply masks in priority order, respecting already-painted pixels
      for (const { maskUrl, className } of classifiedMasks) {
        if (className === 'Background') continue;

        const classInfo = ANNOTATION_CLASSES.find(c => c.name === className);
        if (!classInfo) continue;

        const maskImg = await loadImage(maskUrl);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = overlayCanvas.width;
        tempCanvas.height = overlayCanvas.height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

        const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
        const pixels = currentData.data;
        const classColor = hexToRgb(classInfo.color);

        let applied = false;
        for (let j = 0; j < maskData.data.length; j += 4) {
          const pixelIndex = j / 4;
          // Only paint if mask is active AND pixel isn't already painted
          if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
            pixels[j] = classColor.r;
            pixels[j + 1] = classColor.g;
            pixels[j + 2] = classColor.b;
            pixels[j + 3] = 255;
            paintedPixels.add(pixelIndex); // Mark as painted
            applied = true;
          }
        }

        if (applied) {
          ctx.putImageData(currentData, 0, 0);
          counts[className] = (counts[className] || 0) + 1;
        }
      }

      saveToHistory();

      const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ');

      // Add completion message to chat
      setAiMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ **Segmentation Complete!**\n\nApplied regions:\n${summary ? summary.split(', ').map(s => `• ${s}`).join('\n') : '• No golf features found'}\n\nUse the brush tool to refine, or provide feedback to re-analyze.`,
        timestamp: new Date(),
      }]);

      toast.success(`✨ Smart segmentation complete! Found: ${summary || 'no golf features'}. Use brush to refine.`);

    } catch (error: any) {
      console.error('Smart segment error:', error);
      setAiMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ **Segmentation Failed**\n\n${error.message}\n\nTry again or use manual annotation.`,
        timestamp: new Date(),
      }]);
      toast.error(`Smart segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Segment a single class based on color analysis
  const handleSegmentClass = async (targetClass: typeof ANNOTATION_CLASSES[0]) => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    toast.info(`🎯 SAM 2 finding ${targetClass.name} regions...`);

    try {
      // Step 1: Get all masks from SAM 2
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: targetClass.name }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions. Try a different image.');
        return;
      }

      toast.info(`🔍 Analyzing ${individualMasks.length} regions for ${targetClass.name}...`);

      // Get original image data for color analysis
      const baseCtx = baseCanvas.getContext('2d')!;
      const originalImageData = baseCtx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing annotation data to preserve already-painted areas
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();

      // Mark already painted pixels AND check if this class was already used
      const targetColor = hexToRgb(targetClass.color);
      let sameClassPixelCount = 0;

      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0 || existingData.data[i] > 0 || existingData.data[i + 1] > 0 || existingData.data[i + 2] > 0) {
          paintedPixels.add(i / 4);
          // Check if this pixel is already our target class color
          if (Math.abs(existingData.data[i] - targetColor.r) < 5 &&
              Math.abs(existingData.data[i + 1] - targetColor.g) < 5 &&
              Math.abs(existingData.data[i + 2] - targetColor.b) < 5) {
            sameClassPixelCount++;
          }
        }
      }

      // If we already have some of this class, use more lenient detection (second pass)
      const isSecondPass = sameClassPixelCount > 1000;
      if (isSecondPass) {
        toast.info(`🔄 Second pass for ${targetClass.name} - using broader detection...`);
      }

      let appliedCount = 0;
      let analyzedCount = 0;

      // Process ALL masks for thorough analysis
      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const colors = analyzeColorsInMask(originalImageData, maskData);
          const detectedClass = classifyByColor(colors, targetClass.name, isSecondPass);

          analyzedCount++;

          // Only apply if it matches the target class
          if (detectedClass !== targetClass.name) continue;

          const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
          const pixels = currentData.data;
          const classColor = hexToRgb(targetClass.color);

          let applied = false;
          for (let j = 0; j < maskData.data.length; j += 4) {
            const pixelIndex = j / 4;
            // Only paint if mask is active AND pixel isn't already painted
            if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
              pixels[j] = classColor.r;
              pixels[j + 1] = classColor.g;
              pixels[j + 2] = classColor.b;
              pixels[j + 3] = 255;
              paintedPixels.add(pixelIndex); // Mark as painted
              applied = true;
            }
          }

          if (applied) {
            ctx.putImageData(currentData, 0, 0);
            appliedCount++;
          }
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      saveToHistory();

      if (appliedCount > 0) {
        const passInfo = isSecondPass ? ' (broader search)' : '';
        toast.success(`✨ Applied ${appliedCount} ${targetClass.name} region${appliedCount > 1 ? 's' : ''}${passInfo}. Use brush to refine.`);
      } else {
        const suggestion = isSecondPass
          ? 'No more found. Try manual annotation.'
          : 'Try clicking again for broader search, or use manual annotation.';
        toast.warning(`No ${targetClass.name} found. ${suggestion}`);
      }

    } catch (error: any) {
      console.error('Segment class error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Helper: Load image as promise
  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = url;
    });
  };

  // Helper: Analyze colors in original image where mask is active
  const analyzeColorsInMask = (
    originalData: ImageData,
    maskData: ImageData
  ): { avgR: number; avgG: number; avgB: number; pixelCount: number } => {
    let totalR = 0, totalG = 0, totalB = 0, count = 0;

    for (let i = 0; i < maskData.data.length; i += 4) {
      if (maskData.data[i] > 128) {
        totalR += originalData.data[i];
        totalG += originalData.data[i + 1];
        totalB += originalData.data[i + 2];
        count++;
      }
    }

    return {
      avgR: count > 0 ? totalR / count : 0,
      avgG: count > 0 ? totalG / count : 0,
      avgB: count > 0 ? totalB / count : 0,
      pixelCount: count
    };
  };

  // Helper: Classify using GPT's custom color profiles for this specific image
  const classifyWithCustomProfiles = (
    colors: { avgR: number; avgG: number; avgB: number; pixelCount: number },
    colorProfiles: { [key: string]: {
      minR: number; maxR: number;
      minG: number; maxG: number;
      minB: number; maxB: number;
      minBrightness: number; maxBrightness: number;
    }}
  ): string => {
    const { avgR, avgG, avgB, pixelCount } = colors;

    // If no custom profiles, fall back to global classification
    if (Object.keys(colorProfiles).length === 0) {
      return classifyByColor(colors);
    }

    // Skip very small regions
    if (pixelCount < 200) return 'Background';

    const brightness = (avgR + avgG + avgB) / 3;

    // Score each class based on how well the color fits GPT's profile
    const scores: { [key: string]: number } = {};
    const validClasses = ['Water', 'Bunker', 'Green', 'Tee', 'Fairway'];

    for (const cls of validClasses) {
      const profile = colorProfiles[cls];
      if (!profile) {
        scores[cls] = 0;
        continue;
      }

      let score = 0;

      // Check if RGB values are within the profile's range (with some tolerance)
      const tolerance = 25; // Allow some flexibility

      const rInRange = avgR >= profile.minR - tolerance && avgR <= profile.maxR + tolerance;
      const gInRange = avgG >= profile.minG - tolerance && avgG <= profile.maxG + tolerance;
      const bInRange = avgB >= profile.minB - tolerance && avgB <= profile.maxB + tolerance;
      const brightInRange = brightness >= profile.minBrightness - tolerance && brightness <= profile.maxBrightness + tolerance;

      // Base score for being in range
      if (rInRange) score += 20;
      if (gInRange) score += 20;
      if (bInRange) score += 20;
      if (brightInRange) score += 20;

      // Bonus for being close to the middle of the range
      const midR = (profile.minR + profile.maxR) / 2;
      const midG = (profile.minG + profile.maxG) / 2;
      const midB = (profile.minB + profile.maxB) / 2;

      const distFromMid = Math.sqrt(
        Math.pow(avgR - midR, 2) + Math.pow(avgG - midG, 2) + Math.pow(avgB - midB, 2)
      );

      // Closer to middle = higher score (max 30 bonus)
      score += Math.max(0, 30 - distFromMid / 3);

      // Size adjustments based on class
      if (cls === 'Green' && pixelCount < 5000) score += 15;
      if (cls === 'Tee' && pixelCount < 2000) score += 20;
      if (cls === 'Fairway' && pixelCount > 5000) score += 15;

      // CRITICAL: Prevent trees from being classified as water
      // Trees are dark GREEN dominant, Water is BLUE dominant
      if (cls === 'Water') {
        // If green is dominant over blue, it's likely trees, NOT water
        if (avgG > avgB + 10) {
          score -= 60; // Heavy penalty - this is probably trees
        }
        // Water MUST have blue as dominant or near-dominant
        if (avgB < avgG && avgB < avgR) {
          score -= 40; // Blue should be significant for water
        }
        // Very dark green = trees, not water
        if (avgG > avgR && avgG > avgB && brightness < 80) {
          score -= 50; // Dark green = trees
        }
      }

      scores[cls] = score;
    }

    // Find best match
    let bestClass = 'Background';
    let bestScore = 40; // Minimum threshold

    for (const [cls, score] of Object.entries(scores)) {
      if (score > bestScore) {
        bestScore = score;
        bestClass = cls;
      }
    }

    // If no good match from custom profiles, fall back to global classification
    if (bestClass === 'Background') {
      return classifyByColor(colors);
    }

    return bestClass;
  };

  // Helper: Classify region by average color with target class focus (global fallback)
  const classifyByColor = (
    colors: { avgR: number; avgG: number; avgB: number; pixelCount: number },
    targetClass?: string,
    isSecondPass?: boolean
  ): string => {
    const { avgR, avgG, avgB, pixelCount } = colors;

    // On second pass, be more lenient with size requirements
    const minPixels = isSecondPass ? 150 : (targetClass === 'Tee' ? 200 : 300);
    if (pixelCount < minPixels) return 'Background';

    // Second pass multiplier - boost target class score more aggressively
    const secondPassBoost = isSecondPass ? 2.0 : 1.0;

    // Calculate useful color metrics
    const brightness = (avgR + avgG + avgB) / 3;
    const greenDominance = avgG - Math.max(avgR, avgB);
    const blueDominance = avgB - Math.max(avgR, avgG);
    const saturation = Math.max(avgR, avgG, avgB) - Math.min(avgR, avgG, avgB);

    // Score each class based on how well colors match
    const scores: { [key: string]: number } = {
      'Water': 0,
      'Bunker': 0,
      'Green': 0,
      'Tee': 0,
      'Fairway': 0,
    };

    // WATER: Blue tones - MUST have blue dominant, NOT green dominant (that's trees!)
    // Key: Water has blue >= green, Trees have green > blue
    if (avgB > avgR && avgB >= avgG - 5) {
      scores['Water'] += 30;
      if (avgB > avgG) scores['Water'] += 25; // Blue clearly dominant = likely water
      if (avgB > 80) scores['Water'] += 10;
      // Penalize if green is dominant - that's trees, not water!
      if (avgG > avgB + 15) scores['Water'] -= 50; // Green dominant = trees
      if (avgG > avgR && avgG > avgB && brightness < 80) scores['Water'] -= 60; // Dark green = definitely trees
    }
    // Extra penalty for dark green areas being classified as water
    if (avgG > avgB + 10 && avgG > avgR && brightness < 90) {
      scores['Water'] -= 40; // This is trees, not water
    }

    // BUNKER: Sandy/tan colors - high R and G, lower B, not too bright
    if (avgR > 140 && avgG > 120 && avgB < avgG + 20) {
      scores['Bunker'] += 25;
      if (avgR > avgB + 30) scores['Bunker'] += 20;
      if (avgG > avgB + 20) scores['Bunker'] += 15;
      if (brightness > 140 && brightness < 220) scores['Bunker'] += 15;
      // Sandy look: R and G close together, both higher than B
      if (Math.abs(avgR - avgG) < 40 && avgR > avgB) scores['Bunker'] += 15;
    }

    // GREEN (putting green): Small, bright green areas
    // KEY DISTINCTION: Greens are SMALL (typically under 5000 pixels)
    if (greenDominance > 10 && avgG > 100 && brightness > 95) {
      // Base score for green-ish color
      scores['Green'] += 15;
      if (avgG > 120) scores['Green'] += 10;
      if (greenDominance > 20) scores['Green'] += 10;
      if (saturation > 40) scores['Green'] += 10;

      // SIZE IS KEY: Greens are small!
      if (pixelCount < 3000) scores['Green'] += 40;  // Very likely a green
      else if (pixelCount < 6000) scores['Green'] += 25;
      else if (pixelCount < 10000) scores['Green'] += 5;
      else scores['Green'] -= 30; // Too big = probably fairway, not green
    }

    // TEE: Very small bright green areas
    if (greenDominance > 8 && avgG > 95 && brightness > 90) {
      scores['Tee'] += 10;
      // Tees are VERY small
      if (pixelCount < 1500) scores['Tee'] += 45;
      else if (pixelCount < 2500) scores['Tee'] += 30;
      else if (pixelCount < 4000) scores['Tee'] += 10;
      else scores['Tee'] -= 25; // Too big for a tee
    }

    // FAIRWAY: Green areas that aren't small enough to be greens/tees
    // More lenient on size, focus on "green but not tiny"
    if (avgG > avgR && avgG > avgB && brightness > 75 && brightness < 150) {
      scores['Fairway'] += 20; // Base score for any greenish area
      if (avgG > 80) scores['Fairway'] += 10;
      if (greenDominance > 3) scores['Fairway'] += 10;

      // Size bonuses (but not as strict)
      if (pixelCount > 10000) scores['Fairway'] += 25; // Large = likely fairway
      else if (pixelCount > 5000) scores['Fairway'] += 15;
      else if (pixelCount > 3000) scores['Fairway'] += 10;
      // Small penalty but not too harsh
      else if (pixelCount < 2000) scores['Fairway'] -= 10;

      // Penalize if too dark (likely trees)
      if (brightness < 70) scores['Fairway'] -= 35;
    }

    // TREES/ROUGH penalty: Dark green areas should be Background
    if (brightness < 65 && avgG > avgR) {
      scores['Fairway'] -= 40;
      scores['Green'] -= 25;
      scores['Tee'] -= 25;
    }

    // If targeting a specific class, boost its score to be more inclusive
    // On second pass, boost even more aggressively to find borderline matches
    if (targetClass && scores[targetClass] !== undefined) {
      const boost = isSecondPass ? 2.5 : 1.5;
      scores[targetClass] *= boost;
    }

    // On second pass, also lower the minimum threshold to catch more
    const minThreshold = isSecondPass ? 15 : 25;

    // Find the highest scoring class
    let bestClass = 'Background';
    let bestScore = minThreshold; // Minimum threshold to be classified

    for (const [className, score] of Object.entries(scores)) {
      if (score > bestScore) {
        bestScore = score;
        bestClass = className;
      }
    }

    return bestClass;
  };

  // Helper: Check if color is compatible with a class (sanity check for GPT suggestions)
  const isColorCompatible = (
    colors: { avgR: number; avgG: number; avgB: number; pixelCount: number },
    className: string
  ): boolean => {
    const { avgR, avgG, avgB } = colors;
    const brightness = (avgR + avgG + avgB) / 3;

    switch (className) {
      case 'Water':
        // Water MUST have blue dominant or near-dominant, NOT green dominant (that's trees!)
        // Reject if green is clearly dominant over blue
        if (avgG > avgB + 15) return false; // Green dominant = trees, not water
        if (avgG > avgR && avgG > avgB && brightness < 85) return false; // Dark green = trees
        return avgB > avgR - 20 && avgB >= avgG - 10; // Blue should be significant
      case 'Bunker':
        // Bunker should be tan/sandy (not green or blue dominant)
        return avgR > 100 && avgG > 90 && avgB < avgG + 40;
      case 'Green':
      case 'Tee':
      case 'Fairway':
        // These MUST be green - G must be dominant, not sandy/tan
        // Reject if it looks like a bunker (R and G high, similar values, B lower)
        const isSandy = avgR > 140 && avgG > 120 && Math.abs(avgR - avgG) < 50 && avgB < avgG;
        if (isSandy) return false;
        // Must have green as the dominant or near-dominant color
        return avgG > avgR - 10 && avgG > avgB && brightness > 60;
      default:
        return true;
    }
  };

  // Helper: Priority for layering (lower = applied first, won't be overwritten)
  const getClassPriority = (className: string): number => {
    const priorities: { [key: string]: number } = {
      'Water': 1,
      'Bunker': 2,
      'Green': 3,
      'Tee': 4,
      'Fairway': 5,
      'Background': 99
    };
    return priorities[className] || 99;
  };

  // Helper: Convert hex color to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  };

  // Handle save
  const handleSave = () => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    // Create a composite canvas with background fill
    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = overlayCanvas.width;
    compositeCanvas.height = overlayCanvas.height;
    const ctx = compositeCanvas.getContext('2d')!;

    // Fill entire canvas with background color (everything unannotated = background)
    ctx.fillStyle = ANNOTATION_CLASSES[0].color; // Background color
    ctx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);

    // Draw the user's annotations on top
    ctx.drawImage(overlayCanvas, 0, 0);

    // Export the composite
    const annotationData = compositeCanvas.toDataURL('image/png');
    onSave(annotationData);
    toast.success('Annotation saved! Unannotated areas = Background');
  };

  // AI Agent annotation
  const handleAiAnnotate = async () => {
    setIsAiLoading(true);
    toast.info('🤖 AI Agent analyzing the course (GPT-4 Vision + SAM 2)...');

    try {
      const response = await fetch('/api/ai-annotate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ imageData }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to get AI annotation');
      }

      const data = await response.json();
      const regions = data.regions || [];
      const usedSAM = data.usedSAM;
      const combinedMask = data.combinedMask;

      if (regions.length === 0) {
        toast.warning('AI could not identify any golf course features in this image.');
        return;
      }

      // Draw the AI-generated regions on the canvas
      const overlayCanvas = overlayCanvasRef.current;
      if (!overlayCanvas) return;

      const ctx = overlayCanvas.getContext('2d');
      if (!ctx) return;

      // Clear existing annotations
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Map class names to our annotation classes
      const classMap: { [key: string]: typeof ANNOTATION_CLASSES[0] } = {
        'fairway': ANNOTATION_CLASSES[1],
        'green': ANNOTATION_CLASSES[2],
        'tee': ANNOTATION_CLASSES[3],
        'bunker': ANNOTATION_CLASSES[4],
        'water': ANNOTATION_CLASSES[5],
      };

      // If SAM was used and we have a combined mask, draw it
      if (usedSAM && combinedMask) {
        toast.info('🎯 SAM 2 generated precise masks, applying...');

        // Load the SAM mask image
        const maskImg = new Image();
        maskImg.crossOrigin = 'anonymous';

        await new Promise<void>((resolve, reject) => {
          maskImg.onload = () => {
            // Draw the mask - SAM returns a binary mask
            // We need to colorize it based on feature classes
            ctx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);
            resolve();
          };
          maskImg.onerror = () => reject(new Error('Failed to load SAM mask'));
          maskImg.src = combinedMask;
        });

        // Also draw colored circles at each feature point to show classifications
        regions.forEach((region: any) => {
          const classInfo = classMap[region.class.toLowerCase()];
          if (!classInfo || !region.point) return;

          const x = (region.point.x / 100) * overlayCanvas.width;
          const y = (region.point.y / 100) * overlayCanvas.height;

          // Draw a colored dot at the feature point
          ctx.fillStyle = classInfo.color;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        });

      } else {
        // Fallback: Draw polygons or points without SAM
        regions.forEach((region: any) => {
          const classInfo = classMap[region.class.toLowerCase()];
          if (!classInfo) return;

          ctx.fillStyle = classInfo.color;
          ctx.globalAlpha = 0.6;

          if (region.polygon && region.polygon.length > 0) {
            // Draw polygon
            ctx.beginPath();
            region.polygon.forEach((point: number[], index: number) => {
              const x = (point[0] / 100) * overlayCanvas.width;
              const y = (point[1] / 100) * overlayCanvas.height;
              if (index === 0) {
                ctx.moveTo(x, y);
              } else {
                ctx.lineTo(x, y);
              }
            });
            ctx.closePath();
            ctx.fill();
          } else if (region.point) {
            // Draw a circle at the point
            const x = (region.point.x / 100) * overlayCanvas.width;
            const y = (region.point.y / 100) * overlayCanvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 20, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.globalAlpha = 1.0;
        });
      }

      // Save to history
      saveToHistory();

      const method = usedSAM ? 'GPT-4 Vision + SAM 2' : 'GPT-4 Vision';
      toast.success(`✨ AI annotation complete using ${method}! Found ${regions.length} features. You can refine manually.`);

    } catch (error: any) {
      console.error('AI annotation error:', error);
      toast.error(`Failed to generate AI annotation: ${error.message}`);
    } finally {
      setIsAiLoading(false);
    }
  };

  // Mouse wheel zoom listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Number keys for class selection
      const num = parseInt(e.key);
      if (num >= 1 && num <= 6) {
        setSelectedClass(ANNOTATION_CLASSES[num - 1]);
        setIsEraser(false);
        return;
      }

      // Zoom shortcuts
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        setZoom(prev => Math.min(400, prev + 25));
        return;
      }
      if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        setZoom(prev => Math.max(50, prev - 25));
        return;
      }
      if (e.key === '0' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setZoom(100);
        setPanOffset({ x: 0, y: 0 });
        return;
      }

      // Shortcuts
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z') {
          e.preventDefault();
          if (e.shiftKey) {
            handleRedo();
          } else {
            handleUndo();
          }
        }
      }

      if (e.key === 'e' || e.key === 'E') {
        setIsEraser(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  return (
    <div className="fixed inset-0 bg-black/95 z-50 flex flex-col">
      {/* Header */}
      <div className="bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-white text-xl">Annotation Tool</h2>
          <p className="text-slate-400 text-sm">Draw to annotate golf course features</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            className="bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-600"
          >
            <Save className="size-4 mr-2" />
            Save Annotation
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="size-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden relative">
        {/* Sidebar Controls */}
        <div className="w-80 bg-slate-900 border-r border-slate-700 p-6 overflow-y-auto">
          {/* Per-Class Segment Buttons */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">AI Segment by Class</Label>
            <p className="text-slate-400 text-xs mb-3">
              Click twice for broader search. Won't overwrite existing.
            </p>
            <div className="grid grid-cols-2 gap-2">
              {ANNOTATION_CLASSES.filter(cls => cls.name !== 'Background').map((cls) => (
                <Button
                  key={cls.name}
                  onClick={() => handleSegmentClass(cls)}
                  disabled={isSamProcessing}
                  className="h-10 text-xs font-medium disabled:opacity-50"
                  style={{
                    backgroundColor: cls.color,
                    color: cls.name === 'Bunker' ? '#000' : '#fff',
                  }}
                >
                  {isSamProcessing ? (
                    <div className="size-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <>
                      <Sparkles className="size-3 mr-1" />
                      {cls.name}
                    </>
                  )}
                </Button>
              ))}
            </div>
            {/* Segment All button */}
            <Button
              onClick={handleSmartSegmentAll}
              disabled={isSamProcessing}
              className="w-full h-12 mt-3 text-sm font-medium disabled:opacity-50 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              {isSamProcessing ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing all classes...
                </>
              ) : (
                <>
                  <Sparkles className="size-4 mr-2" />
                  Segment All Classes
                </>
              )}
            </Button>
            <p className="text-slate-400 text-xs mt-2 text-center">
              GPT-4 + SAM 2 segments all features with smart priority
            </p>

            {/* Segment Everything button */}
            <Button
              onClick={handleSegmentEverything}
              disabled={isSamProcessing}
              className="w-full h-10 mt-3 text-sm font-medium disabled:opacity-50 bg-slate-600 hover:bg-slate-500"
            >
              {isSamProcessing ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Paintbrush className="size-4 mr-2" />
                  Segment Everything
                </>
              )}
            </Button>
            <p className="text-slate-500 text-xs mt-1 text-center">
              SAM 2 segments all regions (uses selected class color)
            </p>
          </div>

          {/* Class Selection */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">Select Class</Label>
            <div className="bg-blue-950/30 border border-blue-600/30 rounded p-2 mb-3">
              <p className="text-xs text-blue-300">
                💡 Tip: Only paint features. Unannotated areas automatically become Background!
              </p>
            </div>
            <div className="space-y-2">
              {ANNOTATION_CLASSES.map((cls, idx) => (
                <button
                  key={cls.name}
                  onClick={() => {
                    setSelectedClass(cls);
                    setIsEraser(false);
                  }}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                    selectedClass.name === cls.name && !isEraser
                      ? 'bg-slate-700 ring-2 ring-emerald-500'
                      : idx === 0
                      ? 'bg-slate-800/50 hover:bg-slate-700/50 opacity-60'
                      : 'bg-slate-800 hover:bg-slate-700'
                  }`}
                >
                  <div
                    className="size-6 rounded border-2 border-slate-600"
                    style={{ backgroundColor: cls.color }}
                  />
                  <div className="flex-1 text-left">
                    <span className="text-white">{cls.name}</span>
                    {idx === 0 && (
                      <span className="text-xs text-slate-500 block">Auto-filled</span>
                    )}
                  </div>
                  <span className="text-slate-500 text-sm">{cls.key}</span>
                </button>
              ))}
              
              {/* Eraser */}
              <button
                onClick={() => setIsEraser(true)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                  isEraser
                    ? 'bg-slate-700 ring-2 ring-red-500'
                    : 'bg-slate-800 hover:bg-slate-700'
                }`}
              >
                <div className="size-6 rounded border-2 border-slate-600 bg-slate-900 flex items-center justify-center">
                  <Eraser className="size-4 text-slate-400" />
                </div>
                <span className="text-white flex-1 text-left">Eraser</span>
                <span className="text-slate-500 text-sm">E</span>
              </button>
            </div>
          </div>

          {/* Brush Size */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Brush Size: {brushSize}px
            </Label>
            <Slider
              value={[brushSize]}
              onValueChange={(values) => setBrushSize(values[0])}
              min={5}
              max={100}
              step={5}
              className="mb-2"
            />
          </div>

          {/* Overlay Opacity */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Overlay Opacity: {opacity}%
            </Label>
            <Slider
              value={[opacity]}
              onValueChange={(values) => setOpacity(values[0])}
              min={0}
              max={100}
              step={5}
            />
          </div>

          {/* Zoom */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Zoom: {zoom}%
            </Label>
            <Slider
              value={[zoom]}
              onValueChange={(values) => setZoom(values[0])}
              min={50}
              max={400}
              step={10}
            />
            <div className="flex gap-2 mt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setZoom(Math.max(50, zoom - 25))}
                className="flex-1"
              >
                -
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setZoom(100);
                  setPanOffset({ x: 0, y: 0 });
                }}
                className="flex-1"
              >
                Reset
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setZoom(Math.min(400, zoom + 25))}
                className="flex-1"
              >
                +
              </Button>
            </div>
          </div>

          {/* Tools */}
          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowOverlay(!showOverlay)}
              className="w-full"
            >
              {showOverlay ? (
                <>
                  <EyeOff className="size-4 mr-2" />
                  Hide Overlay
                </>
              ) : (
                <>
                  <Eye className="size-4 mr-2" />
                  Show Overlay
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleUndo}
              disabled={historyIndex <= 0}
              className="w-full"
            >
              <Undo2 className="size-4 mr-2" />
              Undo (Ctrl+Z)
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleRedo}
              disabled={historyIndex >= history.length - 1}
              className="w-full"
            >
              <Redo2 className="size-4 mr-2" />
              Redo (Ctrl+Shift+Z)
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleClear}
              className="w-full text-red-400 border-red-400 hover:bg-red-950"
            >
              <X className="size-4 mr-2" />
              Clear All
            </Button>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="mt-6 pt-6 border-t border-slate-700">
            <Label className="text-white mb-3 block">Controls</Label>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="flex justify-between">
                <span>1-6</span>
                <span>Select class</span>
              </div>
              <div className="flex justify-between">
                <span>E</span>
                <span>Toggle eraser</span>
              </div>
              <div className="flex justify-between">
                <span>Scroll</span>
                <span>Zoom</span>
              </div>
              <div className="flex justify-between">
                <span>Middle Mouse</span>
                <span>Pan</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+0</span>
                <span>Reset view</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+Z</span>
                <span>Undo</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+Shift+Z</span>
                <span>Redo</span>
              </div>
            </div>
          </div>
        </div>

        {/* Canvas Area */}
        <div
          ref={containerRef}
          className="flex-1 flex items-center justify-center overflow-hidden bg-slate-950 relative"
        >
          <div
            className="relative transition-transform duration-100 ease-out"
            style={{
              cursor: isPanning ? 'grabbing' : 'none',
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom / 100})`,
              transformOrigin: 'center',
            }}
          >
            {/* Base image canvas */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0"
              style={{ imageRendering: 'auto' }}
            />

            {/* Annotation overlay canvas */}
            <canvas
              ref={overlayCanvasRef}
              className="relative"
              style={{
                opacity: showOverlay ? opacity / 100 : 0,
                imageRendering: 'auto',
                cursor: 'none',
              }}
              onMouseDown={handleCanvasMouseDown}
              onMouseUp={handleCanvasMouseUp}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => {
                stopDrawing();
                setIsHovering(false);
                setIsPanning(false);
              }}
              onMouseEnter={() => setIsHovering(true)}
            />
          </div>

          {/* Brush cursor indicator */}
          {isHovering && !isPanning && (
            <div
              className="fixed pointer-events-none rounded-full border-2 border-white mix-blend-difference transition-all duration-75"
              style={{
                width: brushSize * (zoom / 100),
                height: brushSize * (zoom / 100),
                transform: 'translate(-50%, -50%)',
                left: cursorPos.x,
                top: cursorPos.y,
                backgroundColor: isEraser ? 'rgba(255,0,0,0.2)' : `${selectedClass.color}40`,
              }}
            />
          )}

          {/* Zoom indicator */}
          {zoom !== 100 && (
            <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded-lg text-sm">
              {zoom}%
            </div>
          )}

          {/* Pan hint */}
          {zoom > 100 && !isPanning && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-slate-300 px-3 py-1 rounded-lg text-xs">
              Scroll to zoom • Middle mouse to pan
            </div>
          )}
        </div>

        {/* AI Assistant Panel */}
        <div className={`${showAiPanel ? 'w-80' : 'w-0'} transition-all duration-300 bg-slate-900 border-l border-slate-700 flex flex-col overflow-hidden`}>
          {showAiPanel && (
            <>
              {/* Panel Header */}
              <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Bot className="size-5 text-purple-400" />
                  <span className="text-white font-medium">AI Assistant</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAiPanel(false)}
                  className="h-8 w-8 p-0 text-slate-400 hover:text-white"
                >
                  <ChevronRight className="size-4" />
                </Button>
              </div>

              {/* Chat Messages */}
              <div
                ref={chatScrollRef}
                className="flex-1 overflow-y-auto p-4 space-y-4"
              >
                {aiMessages.length === 0 ? (
                  <div className="text-center py-8">
                    <Bot className="size-12 mx-auto mb-3 text-slate-600" />
                    <p className="text-slate-400 text-sm">Click "Analyze" to get AI suggestions</p>
                    <p className="text-slate-500 text-xs mt-1">GPT-4 will analyze the image and suggest color profiles</p>
                  </div>
                ) : (
                  aiMessages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[90%] rounded-lg p-3 ${
                          msg.role === 'user'
                            ? 'bg-purple-600 text-white'
                            : 'bg-slate-800 text-slate-200'
                        }`}
                      >
                        {msg.role === 'assistant' && (
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <Bot className="size-3.5 text-purple-400" />
                            <span className="text-xs text-purple-400 font-medium">GPT-4</span>
                          </div>
                        )}
                        <div className="text-sm whitespace-pre-wrap">
                          {msg.content.split('\n').map((line, i) => {
                            if (line.startsWith('**') && line.endsWith('**')) {
                              return <p key={i} className="font-semibold text-white">{line.replace(/\*\*/g, '')}</p>;
                            }
                            return <p key={i}>{line}</p>;
                          })}
                        </div>
                        {/* Show detected features as color pills */}
                        {msg.colorProfiles && Object.keys(msg.colorProfiles).length > 0 && (
                          <div className="mt-3 pt-3 border-t border-slate-700">
                            <p className="text-xs text-slate-400 mb-2">Detected classes:</p>
                            <div className="flex flex-wrap gap-1.5">
                              {Object.keys(msg.colorProfiles).map((cls) => {
                                const classInfo = ANNOTATION_CLASSES.find(c => c.name === cls);
                                return (
                                  <span
                                    key={cls}
                                    className="px-2 py-1 rounded text-xs font-medium"
                                    style={{
                                      backgroundColor: classInfo?.color || '#666',
                                      color: cls === 'Bunker' ? '#000' : '#fff',
                                    }}
                                  >
                                    {cls}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        <div className="text-[10px] text-slate-500 mt-2">
                          {msg.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))
                )}
                {isAnalyzing && (
                  <div className="flex justify-start">
                    <div className="bg-slate-800 rounded-lg p-3">
                      <div className="flex items-center gap-2">
                        <div className="size-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                        <span className="text-slate-300 text-sm">Analyzing image...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="p-4 border-t border-slate-700 space-y-3">
                <div className="flex gap-2">
                  <Button
                    onClick={() => handleAiAnalyze()}
                    disabled={isAnalyzing || isSamProcessing}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 text-white"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="size-4 mr-2" />
                        Analyze
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={handleApplyAnalysis}
                    disabled={!currentAnalysis || isSamProcessing || isAnalyzing}
                    className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-50"
                  >
                    {isSamProcessing ? (
                      <>
                        <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                        Applying...
                      </>
                    ) : (
                      <>
                        <Check className="size-4 mr-2" />
                        Apply
                      </>
                    )}
                  </Button>
                </div>

                {/* Feedback Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={userFeedback}
                    onChange={(e) => setUserFeedback(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && userFeedback.trim()) {
                        handleAiAnalyze(userFeedback);
                      }
                    }}
                    placeholder="Give feedback to refine..."
                    className="flex-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-purple-500"
                  />
                  <Button
                    onClick={() => userFeedback.trim() && handleAiAnalyze(userFeedback)}
                    disabled={!userFeedback.trim() || isAnalyzing}
                    className="px-3 bg-slate-700 hover:bg-slate-600"
                  >
                    <Send className="size-4" />
                  </Button>
                </div>
                <p className="text-xs text-slate-500 text-center">
                  e.g., "focus more on fairways" or "ignore the trees"
                </p>
              </div>
            </>
          )}
        </div>

        {/* Toggle AI Panel Button (when collapsed) */}
        {!showAiPanel && (
          <Button
            onClick={() => setShowAiPanel(true)}
            className="absolute right-4 top-1/2 -translate-y-1/2 bg-purple-600 hover:bg-purple-700 h-24 w-8 rounded-l-lg rounded-r-none flex items-center justify-center"
          >
            <div className="flex flex-col items-center gap-1">
              <MessageSquare className="size-4" />
              <ChevronLeft className="size-3" />
            </div>
          </Button>
        )}
      </div>

      {/* Status Bar */}
      <div className="bg-slate-900 border-t border-slate-700 px-6 py-2 flex items-center justify-between text-sm">
        <div className="text-slate-400">
          {isEraser ? (
            <span className="text-red-400">Eraser Mode</span>
          ) : (
            <span>
              Drawing: <span style={{ color: selectedClass.color }}>{selectedClass.name}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div className="text-slate-500">
            Zoom: {zoom}%
          </div>
          <div className="text-slate-500">
            Canvas: {canvasRef.current?.width || 0} × {canvasRef.current?.height || 0} px
          </div>
        </div>
      </div>
    </div>
  );
}
