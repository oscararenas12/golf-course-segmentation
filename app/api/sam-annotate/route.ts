import { NextRequest, NextResponse } from 'next/server';
import Replicate from 'replicate';

export async function POST(request: NextRequest) {
  try {
    // Check for API key
    if (!process.env.REPLICATE_API_TOKEN) {
      return NextResponse.json(
        { error: 'Replicate API token not configured. Please set REPLICATE_API_TOKEN in .env.local' },
        { status: 500 }
      );
    }

    const replicate = new Replicate({
      auth: process.env.REPLICATE_API_TOKEN,
    });

    const { imageData, className } = await request.json();

    if (!imageData) {
      return NextResponse.json(
        { error: 'No image data provided' },
        { status: 400 }
      );
    }

    console.log(`SAM 2: Auto-segmenting for class "${className}"`);

    // Create prediction and poll for result
    const prediction = await replicate.predictions.create({
      version: "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
      input: {
        image: imageData,
        points_per_side: 32,
        pred_iou_thresh: 0.86,
        stability_score_thresh: 0.92,
        use_m2m: true
      }
    });

    console.log('SAM 2 prediction created:', prediction.id);

    // Poll until complete
    let result = prediction;
    while (result.status !== 'succeeded' && result.status !== 'failed') {
      await new Promise(resolve => setTimeout(resolve, 2000));
      result = await replicate.predictions.get(prediction.id);
      console.log('SAM 2 prediction status:', result.status);
    }

    if (result.status === 'failed') {
      throw new Error(result.error || 'SAM 2 prediction failed');
    }

    const output = result.output as any;

    console.log('SAM 2 response type:', typeof output);
    console.log('SAM 2 response keys:', output ? Object.keys(output) : 'null');
    console.log('SAM 2 full response:', JSON.stringify(output).substring(0, 500));

    // SAM 2 returns { combined_mask: url or object, individual_masks: [url, ...] }
    let maskUrl = null;
    let allMasks: string[] = [];

    if (typeof output === 'string') {
      maskUrl = output;
    } else if (output) {
      // combined_mask might be a string URL or an object
      if (typeof output.combined_mask === 'string') {
        maskUrl = output.combined_mask;
      } else if (output.combined_mask?.url) {
        maskUrl = output.combined_mask.url;
      }

      // individual_masks (not masks)
      const masks = output.individual_masks || output.masks || [];
      if (Array.isArray(masks)) {
        allMasks = masks.map((m: any) => typeof m === 'string' ? m : m?.url).filter(Boolean);
      }

      // If no combined mask, use first individual mask
      if (!maskUrl && allMasks.length > 0) {
        maskUrl = allMasks[0];
      }
    }

    console.log(`SAM 2 found ${allMasks.length} individual masks, combined URL: ${maskUrl || 'none'}`);

    return NextResponse.json({
      success: true,
      maskUrl,
      allMasks,
      className
    });

  } catch (error: any) {
    console.error('SAM annotation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate SAM annotation', details: error.message },
      { status: 500 }
    );
  }
}
