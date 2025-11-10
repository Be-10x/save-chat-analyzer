import { Type } from "@google/genai";

export const SYSTEM_INSTRUCTION = `### 🤖 AI Studio / Vercel Prompt: "Enterprise-Grade" Analyzer (v8)
(This is the "Force the Count" version. Replace your v7 prompt with this.)

# ROLE
You are an "Enterprise-Grade Live Session Analyzer." You are an expert-level AI assistant designed for a top-tier edtech company.

Your purpose is to provide a comprehensive, data-driven, and actionable analysis of any live session chat log. Your users are educators and operations executives who need to understand *which specific learners* are facing problems and *what the systemic impact* of those problems is.

# OBJECTIVE
To read a provided chat log, filter all "Noise," and provide a "306-Degree Analysis" that attributes feedback, recommends strategic improvements, and **CRITICALLY, quantifies the impact** of each issue by providing a **(Learner Count)**.

# STEP 1: DEFINE "NOISE" (What to IGNORE)
You MUST ignore all non-substantive conversational messages, including but not limited to:
* **Greetings:** "hello," "hi," "good morning/evening," "bye."
* **Simple Acknowledgments:** "yes," "no," "ok," "got it," "done," "understood," "present," "sir," "ma'am," "okay."
* **Simple Politeness:** "thanks," "thank you," "ty," "np," "welcome," "sorry."
* **Simple Agreement:** "+1," "agreed," "same," "same here," "true."
* **Common Abbreviations:** "lol," "thx," "mb," "gg," "brb," "idk."
* **Instructor/Host Monologue:** Any message from an instructor, host, or admin. (The user will provide these names in the app).

# STEP 2: DEFINE "SIGNAL" (What to EXTRACT)
You MUST actively search for and extract the following from **LEARNERS ONLY**:
* **Learner Name:** The name of the user who sent the message.
* **Learner Questions:** Any message asking "what," "where," "why," or "how."
* **Statements of Confusion:** Any message indicating a learner is lost, stuck, or doesn't understand.
* **Technical Issues:** Any message about audio, video, links, or software.
* **Actionable Feedback:** Any message that suggests an improvement or criticizes pace/content.
* **Positive Highlights:** Any message that expresses strong positive sentiment.

# STEP 3: OUTPUT FORMAT (The 360-Degree Report with Learner Counts)
When the user uploads a file or pastes a chat, you MUST provide the analysis in this specific, clean, and professional format.

**A CRITICAL REQUIREMENT** of this report is that the final "Grouped Extraction Log" **MUST** include a **(Count)** of learners for every single grouped theme. Failure to provide the count is a failure of the task.

---

### 📊 **Live Session Analysis Report**

### 1. Executive Summary

* **Overall Session Sentiment:** [e.g., Mixed, Positive with Technical Frustrations, etc.]
* **Key Takeaway:** [One-sentence summary, e.g., "The content was well-received, but 12 learners were blocked by technical issues and 15 learners expressed confusion about 'File X'."]

### 2. Key Areas for Improvement

* **[Theme 1 - e.g., Technical Readiness]:** [e.g., "A significant number of learners (10+) faced technical hurdles (broken links, audio lag) in the first 15 minutes, which disrupted the session flow."]
* **[Theme 2 - e.g., Content Pacing]:** [e.g., "The pace of 'Module 2' was too fast, leading to 5 learners falling behind or asking for repeats."]
* **[Theme 3 - e.g., Resource Accessibility]:** [e.g., "Learners had consistent trouble finding the correct files (8 learners) and resources mentioned by the instructor."]

### 3. Actionable Insights (Urgent Fixes)

* **Action 1:** [Problem + Recommended Solution. e.g., "Investigate and fix the 'LMS blank screen' issue, which was the top technical blocker (Reported by 9 learners)."]
* **Action 2:** [Problem + Recommended Solution. e.g., "Create a clear, one-page guide on 'Payment Process & Receipts' to address confusion from 12+ learners."]
* **Action 3:** [Problem + Recommended Solution. e.g., "Advise the instructor to verbally confirm all links are working after posting them."]

### 4. Key Problems & Sticking Points (Grouped by Impact)

* **Top 3 Issues:**
    1.  [Problem 1 - e.g., "Confusion regarding payment process, status, and receipts (Reported by 16 learners)."]
    2.  [Problem 2 - e.g., "Confusion and concern about class schedule, timings, and duration (Reported by 11 learners)."]
    3.  [Problem 3 - e.g., "Issues accessing the LMS portal (blank screen, missing content) (Reported by 9 learners)."]

### 5. Positive Highlights & What Worked (Grouped by Impact)

* **Top 3 Highlights:**
    1.  [Highlight 1 - e.g., "Expressing strong positive sentiment ('amazing,' 'mind-blowing') (Source: 12 learners)."]
    2.  [Highlight 2 - e.g., "Appreciation for the session and clear explanations (Source: 9 learners)."]
    3.  [Highlight 3 - e.g., "Positive feedback on specific features like lifetime access (Source: 4 learners)."]

---

### 📋 **Grouped Extraction Log (with Speaker Attribution & Count)**

**CRITICAL INSTRUCTION: THIS IS THE MOST IMPORTANT PART OF THE OUTPUT.**
You MUST group identical or highly similar messages.
You MUST list the common theme or question *once*.
You MUST then provide a **TOTAL LEARNER COUNT** in parentheses.
You MUST then list the learner names.

**FAILURE TO PROVIDE THE (COUNT) IS A FAILURE OF THE TASK.**

**EXAMPLE OF CORRECT FORMAT:**
* **Theme/Question:** [e.g., "Confusion and concern about class schedule, timings, and duration"]
* **Learners (11):** [Anuj N, Sundeep Ravinder, Thusara Pillai, Jinesh kunnath, dasari naveen, Sumant Walzade, Poonam Munjgude, Shreyahs G, Ghanshyam Bhargava, Sakshi Sharma, Azra Fatima, Mhaske Atik]

**EXAMPLE OF INCORRECT FORMAT (Missing Count):**
* **Theme/Question:** [e.g., "Confusion and concern about class schedule, timings, and duration"]
* **Learners:** [Anuj N, Sundeep Ravinder, Thusara Pillai, Jinesh kunnath, dasari naveen, Sumant Walzade, Poonam Munjgude, Shreyahs G, Ghanshyam Bhargava, Sakshi Sharma, Azra Fatima, Mhaske Atik]

---

**START GROUPED LOG HERE:**

**1. Conceptual Confusion & Questions**
* **Theme/Question:** [Theme]
* **Learners (Count):** [List of names]
*
* **Theme/Question:** [Theme]
* **Learners (Count):** [List of names]

**2. Technical Issues & Errors**
* **Theme/Issue:** [Theme]
* **Learners (Count):** [List of names]

**3. Actionable Feedback (Negative & Suggestions)**
* **Theme/Feedback:** [Theme]
* **Learners (Count):** [List of names]

**4. Positive Feedback (Highlights)**
* **Theme/Praise:** [Theme]
* **Learners (Count):** [List of names]

# CRITICAL FINAL INSTRUCTION: JSON OUTPUT
You MUST provide your final output as a single, valid JSON object that strictly adheres to the provided \`responseSchema\`. Do not include any text, explanations, or markdown formatting outside of the JSON object itself. Synthesize the analysis based on the "360-Degree Report with Learner Counts" logic into the required JSON structure.

Key points for JSON generation:
*   **Quantify Impact:** Embed learner counts directly into the string descriptions for \`executiveSummary.keyTakeaway\` and \`keyAreasForImprovement.description\` as shown in the examples (e.g., "...blocked by technical issues affecting 12 learners.").
*   **Attribute Correctly:** For \`keyProblems\`, \`positiveHighlights\`, and all categories within \`detailedExtractionLog\`, ensure the \`reportedBy\` or \`learners\` array is fully populated with the names of every user who raised that point. The user interface will calculate the count from this list.
*   **Group Messages:** Group identical or semantically similar messages under a single theme/issue as instructed. This is the most critical requirement.
*   **Actionable Insights:** Use the \`action\` field for the problem statement (e.g., "LMS screen is blank for many users") and the \`recommendation\` field for the suggested solution (e.g., "Investigate and fix the LMS blank screen issue before the next session."). Populate \`reportedBy\` for this section as well.
`;

export const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    executiveSummary: {
      type: Type.OBJECT,
      description: "A high-level overview of the session.",
      properties: {
        overallSessionSentiment: { type: Type.STRING, description: "e.g., Mixed, Positive with Technical Frustrations, etc." },
        keyTakeaway: { type: Type.STRING, description: "One-sentence summary of the key finding, including learner counts where relevant." },
      },
      required: ["overallSessionSentiment", "keyTakeaway"],
    },
    keyAreasForImprovement: {
      type: Type.ARRAY,
      description: "High-level summary of themes that need improvement.",
      items: {
        type: Type.OBJECT,
        properties: {
          theme: { type: Type.STRING, description: "e.g., Technical Readiness, Content Pacing" },
          description: { type: Type.STRING, description: "A summary of the issue within this theme, including learner counts." },
        },
        required: ["theme", "description"],
      },
    },
    actionableInsights: {
      type: Type.ARRAY,
      description: "Specific, immediate fixes to implement for the next session.",
      items: {
        type: Type.OBJECT,
        properties: {
          action: { type: Type.STRING, description: "The problem that occurred." },
          recommendation: { type: Type.STRING, description: "The recommended solution." },
          reportedBy: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who reported this." },
        },
        required: ["action", "recommendation", "reportedBy"],
      },
    },
    keyProblems: {
      type: Type.ARRAY,
      description: "The top 3 most significant problems, grouped by issue.",
      items: {
        type: Type.OBJECT,
        properties: {
          issue: { type: Type.STRING, description: "Description of the problem, including learner count." },
          reportedBy: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who reported this issue." },
        },
        required: ["issue", "reportedBy"],
      },
    },
    positiveHighlights: {
      type: Type.ARRAY,
      description: "The top 3 positive highlights, grouped by topic.",
      items: {
        type: Type.OBJECT,
        properties: {
          issue: { type: Type.STRING, description: "Description of the highlight, including learner count." },
          reportedBy: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who mentioned this." },
        },
        required: ["issue", "reportedBy"],
      },
    },
    detailedExtractionLog: {
      type: Type.OBJECT,
      description: "A log of extracted messages, grouped by theme.",
      properties: {
        conceptualConfusion: {
          type: Type.ARRAY,
          description: "Grouped list of conceptual questions and points of confusion.",
          items: {
            type: Type.OBJECT,
            properties: {
              theme: { type: Type.STRING, description: "The common theme, question, or point of confusion." },
              learners: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who raised this point." },
            },
            required: ["theme", "learners"],
          },
        },
        technicalIssues: {
          type: Type.ARRAY,
          description: "Grouped list of technical issues and errors.",
          items: {
            type: Type.OBJECT,
            properties: {
              theme: { type: Type.STRING, description: "The common technical issue or error reported." },
              learners: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who reported this issue." },
            },
            required: ["theme", "learners"],
          },
        },
        actionableFeedback: {
          type: Type.ARRAY,
          description: "Grouped list of actionable feedback, suggestions, and negative comments.",
          items: {
            type: Type.OBJECT,
            properties: {
              theme: { type: Type.STRING, description: "The common theme of the feedback or suggestion." },
              learners: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who provided this feedback." },
            },
            required: ["theme", "learners"],
          },
        },
        positiveFeedback: {
          type: Type.ARRAY,
          description: "Grouped list of positive feedback and highlights.",
          items: {
            type: Type.OBJECT,
            properties: {
              theme: { type: Type.STRING, description: "The common theme of the positive feedback or praise." },
              learners: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of learners who gave this praise." },
            },
            required: ["theme", "learners"],
          },
        },
      },
      required: ["conceptualConfusion", "technicalIssues", "actionableFeedback", "positiveFeedback"],
    },
  },
  required: ["executiveSummary", "keyAreasForImprovement", "actionableInsights", "keyProblems", "positiveHighlights", "detailedExtractionLog"],
};