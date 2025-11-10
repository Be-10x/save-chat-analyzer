import { GoogleGenAI } from "@google/genai";
import type { AnalysisReportData } from '../types';
import { SYSTEM_INSTRUCTION, RESPONSE_SCHEMA } from '../constants';

export async function analyzeChatLog(chatLog: string, instructorNames: string): Promise<AnalysisReportData> {
  
  // Use the VITE_ prefix to make the key available in the browser
  const ai = new GoogleGenAI({ apiKey: process.env.VITE_API_KEY });

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
        throw new Error(error.message);
    }
    throw new Error("An unknown error occurred while processing the AI response.");
  }
}