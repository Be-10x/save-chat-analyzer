import { GoogleGenAI } from "@google/genai";
import type { AnalysisReportData } from '../types';
import { SYSTEM_INSTRUCTION, RESPONSE_SCHEMA } from '../constants';

// FIX: Use import.meta.env for client-side variables in Vite.
export async function analyzeChatLog(chatLog: string, instructorNames: string): Promise<AnalysisReportData> {
  
  // The API key is retrieved from environment variables as per Vite's standard.
  const apiKey = import.meta.env.VITE_API_KEY;

  if (!apiKey) {
    // This error will be shown if the check in App.tsx is bypassed for some reason.
    throw new Error("VITE_API_KEY is not configured. Please ensure it is set in your environment variables and the project is redeployed.");
  }

  // Create a new instance for each call to ensure the latest key from the environment is used.
  // The SDK internally handles the `process.env.API_KEY` when available, but we explicitly pass it for clarity and consistency in client-side apps.
  const ai = new GoogleGenAI({ apiKey });

  const userPrompt = `
  Here is the chat log. Please analyze it.
  The instructor(s)/host(s) for this session are: ${instructorNames}. Please ignore their messages as per the instructions.

  --- CHAT LOG START ---
  ${chatLog}
  --- CHAT LOG END ---
  `;

  try {
    // Use the non-streaming generateContent method for a single JSON response.
    const response = await ai.models.generateContent({
        model: "gemini-2.5-pro",
        contents: userPrompt,
        config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: "application/json",
            // @ts-ignore - The schema type is compatible
            responseSchema: RESPONSE_SCHEMA,
            thinkingConfig: { thinkingBudget: 32768 }
        },
    });

    // Access the response text directly, no aggregation needed.
    const jsonText = response.text.trim();

    if (!jsonText) {
        throw new Error("The AI returned an empty response. The input might be too complex or contain restricted content.");
    }
    
    const data = JSON.parse(jsonText);
    return data as AnalysisReportData;

  } catch (error) {
    console.error("Error processing Gemini API response:", error);
    if (error instanceof SyntaxError) {
        throw new Error("The AI returned incomplete or malformed data. This can happen with very complex requests or a service interruption. Please try again.");
    }
    if (error instanceof Error) {
        throw error; // Re-throw the original error to be handled by the UI
    }
    throw new Error("An unknown error occurred while processing the AI response.");
  }
}