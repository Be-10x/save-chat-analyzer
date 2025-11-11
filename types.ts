// FIX: Add a triple-slash directive to include Vite client types.
// This resolves the error "Property 'env' does not exist on type 'ImportMeta'"
// by making Vite's `import.meta.env` properties available globally.
/// <reference types="vite/client" />

export interface GroupedMessage {
  theme: string;
  learners: string[];
}

export interface DetailedExtractionLog {
  conceptualConfusion: GroupedMessage[];
  technicalIssues: GroupedMessage[];
  actionableFeedback: GroupedMessage[];
  positiveFeedback: GroupedMessage[];
}

export interface ExecutiveSummary {
  overallSessionSentiment: string;
  keyTakeaway: string;
}

export interface ImprovementArea {
  theme: string;
  description: string;
}

export interface ActionableInsight {
  action: string;
  recommendation: string;
  reportedBy: string[];
}

export interface GroupedPoint {
    issue: string;
    reportedBy: string[];
}

export interface AnalysisReportData {
  executiveSummary: ExecutiveSummary;
  keyAreasForImprovement: ImprovementArea[];
  actionableInsights: ActionableInsight[];
  keyProblems: GroupedPoint[];
  positiveHighlights: GroupedPoint[];
  detailedExtractionLog: DetailedExtractionLog;
}