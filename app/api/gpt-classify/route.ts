import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

export async function POST(request: NextRequest) {
  try {
    const { imageData, feedback } = await request.json();

    if (!imageData) {
      return NextResponse.json({ error: 'No image data provided' }, { status: 400 });
    }

    // Build feedback context if provided
    const feedbackContext = feedback
      ? `\n\nUSER FEEDBACK: The user provided this feedback to refine the analysis: "${feedback}". Please adjust your analysis accordingly.`
      : '';

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: 'OpenAI API key not configured' }, { status: 500 });
    }

    const openai = new OpenAI({ apiKey });

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `You are a golf course aerial imagery analyst. Your job is to:
1. Identify ONLY golf course features - IGNORE everything else (trees, rough, buildings, roads, parking lots, etc.)
2. Analyze the ACTUAL COLORS of golf course features in this specific image

ONLY identify these 5 golf course feature types - nothing else:
- FAIRWAY
- GREEN (putting green)
- TEE
- BUNKER
- WATER

CRITICAL DISTINCTION - GREEN vs FAIRWAY:
- GREEN (putting green): SMALL circular/oval areas (usually 500-5000 sq meters). Located at the END of each hole. Very bright, vibrant green color. Extremely well-maintained, smooth texture. Often surrounded by bunkers. This is where the flag/hole is located.
- FAIRWAY: THE LARGEST green areas in the image - usually 50-70% of the visible golf course area! They look like LARGE BLOBS or elongated patches of medium green grass. Long winding paths connecting tee to green. Medium green color, NOT as bright as putting greens. This is the main playing corridor. IMPORTANT: Fairways often have a CHECKERBOARD or STRIPED pattern from mowing - alternating light/dark green stripes. This striped pattern is a strong indicator of fairway!

SIZE IS THE KEY DIFFERENCE:
- FAIRWAYS are the LARGEST part of the image - big green blobs/patches
- Putting greens are MUCH smaller - tiny circles at the end of fairways
- If it's a large green blob = FAIRWAY (this is the most common feature)
- If it's a small bright circle = GREEN

IGNORE THESE (do NOT include in features or color profiles):
- Trees, forests, wooded areas (dark green/brown)
- Rough grass areas between holes
- Cart paths, roads, parking lots
- Buildings, clubhouse
- Out of bounds areas
- Any non-golf-course terrain

Return a JSON object with:
1. "features" - array of ONLY golf course features found
2. "colorProfiles" - RGB ranges for ONLY the 5 golf feature types visible

Color profile tips:
- Fairway: Medium green, less saturated than greens
- Green: Bright, vibrant green, very saturated
- Bunker: Tan/beige/sand color (varies by course - white sand to brown)
- Water: Blue to dark blue (can be greenish)
- Tee: Similar to green but smaller areas`
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Analyze this golf course image. ONLY identify golf course features (Fairway, Green, Tee, Bunker, Water).

IGNORE: trees, rough, buildings, roads, parking lots, dark wooded areas.

REMEMBER:
- FAIRWAY = THE LARGEST green areas - big blobs covering 50-70% of the golf course! Medium green color. Often has CHECKERBOARD/STRIPED mowing pattern!
- GREEN = SMALL bright circular areas at END of holes (putting surface with flag) - tiny compared to fairways!
- Size is key: Fairways are HUGE blobs, Greens are tiny circles!
- Striped/checkered pattern = FAIRWAY (mowing lines)

Return ONLY valid JSON:

{
  "features": [
    {"class": "Fairway", "centerX": 50, "centerY": 60, "radius": 25},
    {"class": "Green", "centerX": 30, "centerY": 20, "radius": 5}
  ],
  "colorProfiles": {
    "Fairway": {"minR": 60, "maxR": 120, "minG": 80, "maxG": 140, "minB": 40, "maxB": 90, "minBrightness": 70, "maxBrightness": 120},
    "Green": {"minR": 50, "maxR": 100, "minG": 110, "maxG": 170, "minB": 40, "maxB": 90, "minBrightness": 100, "maxBrightness": 150},
    "Bunker": {"minR": 160, "maxR": 230, "minG": 140, "maxG": 210, "minB": 100, "maxB": 180, "minBrightness": 140, "maxBrightness": 200},
    "Water": {"minR": 30, "maxR": 100, "minG": 50, "maxG": 120, "minB": 80, "maxB": 160, "minBrightness": 60, "maxBrightness": 120}
  }
}

Adjust RGB values based on the ACTUAL colors you see in THIS specific image!${feedbackContext}`
            },
            {
              type: 'image_url',
              image_url: {
                url: imageData,
                detail: 'high'
              }
            }
          ]
        }
      ],
      max_tokens: 2500,
    });

    const content = response.choices[0]?.message?.content || '{}';

    // Parse the JSON response
    let parsed: any = { features: [], colorProfiles: {} };
    try {
      // Try to extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      console.error('Failed to parse GPT response:', content);
    }

    // Validate features
    const validFeatures = (parsed.features || [])
      .filter((f: any) =>
        f.class &&
        typeof f.centerX === 'number' &&
        typeof f.centerY === 'number' &&
        ['Fairway', 'Green', 'Tee', 'Bunker', 'Water'].includes(f.class)
      )
      .map((f: any) => ({
        class: f.class,
        centerX: Math.max(0, Math.min(100, f.centerX)),
        centerY: Math.max(0, Math.min(100, f.centerY)),
        radius: f.radius || 10
      }));

    // Validate color profiles
    const colorProfiles: any = {};
    const validClasses = ['Fairway', 'Green', 'Tee', 'Bunker', 'Water'];

    for (const cls of validClasses) {
      if (parsed.colorProfiles && parsed.colorProfiles[cls]) {
        const cp = parsed.colorProfiles[cls];
        colorProfiles[cls] = {
          minR: Math.max(0, Math.min(255, cp.minR || 0)),
          maxR: Math.max(0, Math.min(255, cp.maxR || 255)),
          minG: Math.max(0, Math.min(255, cp.minG || 0)),
          maxG: Math.max(0, Math.min(255, cp.maxG || 255)),
          minB: Math.max(0, Math.min(255, cp.minB || 0)),
          maxB: Math.max(0, Math.min(255, cp.maxB || 255)),
          minBrightness: Math.max(0, Math.min(255, cp.minBrightness || 0)),
          maxBrightness: Math.max(0, Math.min(255, cp.maxBrightness || 255)),
        };
      }
    }

    console.log('GPT-4 identified features:', validFeatures);
    console.log('GPT-4 color profiles:', colorProfiles);

    return NextResponse.json({
      features: validFeatures,
      colorProfiles
    });

  } catch (error: any) {
    console.error('GPT classify error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to classify image' },
      { status: 500 }
    );
  }
}
