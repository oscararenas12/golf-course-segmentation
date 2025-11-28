import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import Replicate from 'replicate';

export async function POST(request: NextRequest) {
  try {
    // Check for API keys
    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: 'OpenAI API key not configured. Please set OPENAI_API_KEY in .env.local' },
        { status: 500 }
      );
    }

    const useSAM = !!process.env.REPLICATE_API_TOKEN;

    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const { imageData } = await request.json();

    if (!imageData) {
      return NextResponse.json(
        { error: 'No image data provided' },
        { status: 400 }
      );
    }

    // Step 1: Use GPT-4 Vision to identify feature CENTER POINTS (not polygons)
    // These points will be used as prompts for SAM 2
    const visionResponse = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `You are analyzing a golf course satellite image to identify features for segmentation.

## Your Task:
Identify the CENTER POINT of each golf course feature. These points will be used to prompt a segmentation model.

## Features to Find (ONLY these 5):
1. **FAIRWAY** - Mowed grass corridors (provide multiple points along long fairways)
2. **GREEN** - Putting surfaces (one point per green, at center)
3. **TEE** - Tee boxes (one point per tee box)
4. **BUNKER** - Sand traps (one point per bunker)
5. **WATER** - Water hazards (one point per water body)

## Instructions:
- For each feature, provide the CENTER point as x,y percentages (0-100)
- For LONG fairways, provide 2-4 points spread along the fairway length
- Be precise - the point should be clearly INSIDE the feature
- Do NOT include background, trees, cart paths, or buildings

## Output Format (JSON only):
{
  "features": [
    {"class": "fairway", "x": 50, "y": 30},
    {"class": "green", "x": 25, "y": 15},
    {"class": "bunker", "x": 28, "y": 18}
  ]
}

Analyze the image and return ONLY the JSON. Be thorough - find ALL features.`,
            },
            {
              type: "image_url",
              image_url: {
                url: imageData,
              },
            },
          ],
        },
      ],
      max_tokens: 4000,
    });

    const content = visionResponse.choices[0].message.content;

    if (!content) {
      return NextResponse.json(
        { error: 'No response from AI' },
        { status: 500 }
      );
    }

    // Parse the GPT-4 Vision response
    let visionData;
    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        visionData = JSON.parse(jsonMatch[0]);
      } else {
        visionData = JSON.parse(content);
      }
    } catch (parseError) {
      console.error('Failed to parse GPT-4 Vision response:', content);
      return NextResponse.json(
        { error: 'Failed to parse AI response', details: content },
        { status: 500 }
      );
    }

    // Filter valid features
    const validClasses = ['fairway', 'green', 'tee', 'bunker', 'water'];
    const features = (visionData.features || []).filter((f: any) =>
      validClasses.includes(f.class?.toLowerCase())
    );

    console.log(`GPT-4 Vision identified ${features.length} feature points`);

    // If no SAM token, fall back to polygon-based response (convert points to simple regions)
    if (!useSAM) {
      console.log('No Replicate token - returning point-based regions without SAM');
      // Convert points to simple circular regions for fallback
      const regions = features.map((f: any) => ({
        class: f.class.toLowerCase(),
        point: { x: f.x, y: f.y },
        // Create a simple polygon around the point as fallback
        polygon: [
          [f.x - 3, f.y - 3],
          [f.x + 3, f.y - 3],
          [f.x + 3, f.y + 3],
          [f.x - 3, f.y + 3],
        ]
      }));
      return NextResponse.json({ regions, usedSAM: false });
    }

    // Step 2: Use SAM 2 to generate precise masks for each feature point
    console.log('Calling SAM 2 for precise segmentation...');

    const replicate = new Replicate({
      auth: process.env.REPLICATE_API_TOKEN,
    });

    // Call SAM 2 with all points at once
    // Group points by class for the response
    const pointCoords = features.map((f: any) => [
      Math.round(f.x * 16.64), // Convert % to pixels (assuming 1664 width)
      Math.round(f.y * 10.24)  // Convert % to pixels (assuming 1024 height)
    ]);
    const pointLabels = features.map(() => 1); // All foreground points

    try {
      const samOutput = await replicate.run(
        "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        {
          input: {
            image: imageData,
            points: pointCoords,
            labels: pointLabels,
          }
        }
      ) as any;

      console.log('SAM 2 response received');

      // Combine GPT-4 class info with SAM masks
      const regions = features.map((f: any, index: number) => ({
        class: f.class.toLowerCase(),
        point: { x: f.x, y: f.y },
        maskUrl: samOutput?.combined_mask || samOutput?.masks?.[index] || null
      }));

      return NextResponse.json({
        regions,
        usedSAM: true,
        combinedMask: samOutput?.combined_mask,
        individualMasks: samOutput?.masks
      });

    } catch (samError: any) {
      console.error('SAM 2 error:', samError);
      // Fall back to point-based response
      const regions = features.map((f: any) => ({
        class: f.class.toLowerCase(),
        point: { x: f.x, y: f.y },
        polygon: [
          [f.x - 3, f.y - 3],
          [f.x + 3, f.y - 3],
          [f.x + 3, f.y + 3],
          [f.x - 3, f.y + 3],
        ]
      }));
      return NextResponse.json({
        regions,
        usedSAM: false,
        samError: samError.message
      });
    }

  } catch (error: any) {
    console.error('AI annotation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate AI annotation', details: error.message },
      { status: 500 }
    );
  }
}
