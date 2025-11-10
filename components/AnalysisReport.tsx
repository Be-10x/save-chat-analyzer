import React from 'react';
import type { AnalysisReportData, GroupedMessage } from '../types';
import { SummaryIcon, ImprovementIcon, ActionIcon, ProblemIcon, HighlightIcon, LogIcon, UserIcon, QuoteIcon } from './icons';

interface AnalysisReportProps {
  report: AnalysisReportData;
}

const ReportSection: React.FC<{ title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => (
  <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-md mb-6">
    <div className="flex items-center mb-4">
      {icon}
      <h3 className="ml-3 text-xl font-bold text-slate-800 dark:text-slate-100">{title}</h3>
    </div>
    <div className="prose prose-slate dark:prose-invert max-w-none">
      {children}
    </div>
  </div>
);

const GroupedMessageCard: React.FC<{ message: GroupedMessage }> = ({ message }) => (
    <div className="p-4 border-l-4 border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700/50 rounded-r-lg not-prose">
        <div className="flex items-start space-x-3">
             <span className="flex-shrink-0 text-slate-400 dark:text-slate-500 mt-1"><QuoteIcon /></span>
             <p className="text-slate-700 dark:text-slate-200 font-semibold">"{message.theme}"</p>
        </div>
        <div className="flex items-start space-x-3 mt-2 pl-8">
            <span className="flex-shrink-0 text-slate-500 dark:text-slate-400"><UserIcon/></span>
            <p className="text-sm text-slate-600 dark:text-slate-300">
                <span className="font-medium">Learners ({message.learners.length}):</span> {message.learners.join(', ')}
            </p>
        </div>
    </div>
);


const AnalysisReport: React.FC<AnalysisReportProps> = ({ report }) => {
  const { 
    executiveSummary, 
    keyAreasForImprovement, 
    actionableInsights, 
    keyProblems, 
    positiveHighlights, 
    detailedExtractionLog 
  } = report;
  
  const renderExtractionLogSection = (title: string, messages: GroupedMessage[]) => (
    messages.length > 0 && (
      <>
        <h4 className="font-semibold mt-6 mb-3 text-lg">{title}</h4>
        <div className="space-y-4">
          {messages.map((msg, index) => <GroupedMessageCard key={index} message={msg} />)}
        </div>
      </>
    )
  );

  return (
    <div className="space-y-8">
       <h2 className="text-3xl font-extrabold text-center text-slate-900 dark:text-white tracking-tight mb-8">📊 Save Chat Analysis Report</h2>

        <ReportSection title="Executive Summary" icon={<SummaryIcon />}>
            <ul className="list-disc pl-5 space-y-2">
                <li><strong>Overall Session Sentiment:</strong> {executiveSummary.overallSessionSentiment}</li>
                <li><strong>Key Takeaway:</strong> {executiveSummary.keyTakeaway}</li>
            </ul>
        </ReportSection>

        <ReportSection title="Key Areas for Improvement" icon={<ImprovementIcon />}>
            <ul className="list-disc pl-5 space-y-3">
                {keyAreasForImprovement.map((item, index) => (
                    <li key={index}><strong>{item.theme}:</strong> {item.description}</li>
                ))}
            </ul>
        </ReportSection>

        <ReportSection title="Actionable Insights (Urgent Fixes)" icon={<ActionIcon />}>
            <ol className="list-decimal pl-5 space-y-3">
                {actionableInsights.map((item, index) => (
                    <li key={index}>
                        <strong>{item.action}: </strong>
                        {item.recommendation}
                        <span className="text-sm text-slate-500 dark:text-slate-400"> (Reported by {item.reportedBy.length}: {item.reportedBy.join(', ')})</span>
                    </li>
                ))}
            </ol>
        </ReportSection>

        <div className="grid md:grid-cols-2 gap-6">
            <ReportSection title="Key Problems & Sticking Points" icon={<ProblemIcon />}>
                 <ol className="list-decimal pl-5 space-y-2">
                    {keyProblems.map((item, index) => (
                        <li key={index}>
                            {item.issue} 
                            <span className="text-sm text-slate-500 dark:text-slate-400"> (Reported by {item.reportedBy.length}: {item.reportedBy.join(', ')})</span>
                        </li>
                    ))}
                </ol>
            </ReportSection>
            <ReportSection title="Positive Highlights & What Worked" icon={<HighlightIcon />}>
                <ol className="list-decimal pl-5 space-y-2">
                    {positiveHighlights.map((item, index) => (
                        <li key={index}>
                            {item.issue} 
                            <span className="text-sm text-slate-500 dark:text-slate-400"> (Source - {item.reportedBy.length}: {item.reportedBy.join(', ')})</span>
                        </li>
                    ))}
                </ol>
            </ReportSection>
        </div>

        <ReportSection title="Detailed Extraction Log" icon={<LogIcon />}>
            {renderExtractionLogSection('Conceptual Confusion & Questions', detailedExtractionLog.conceptualConfusion)}
            {renderExtractionLogSection('Technical Issues & Errors', detailedExtractionLog.technicalIssues)}
            {renderExtractionLogSection('Actionable Feedback (Negative & Suggestions)', detailedExtractionLog.actionableFeedback)}
            {renderExtractionLogSection('Positive Feedback (Highlights)', detailedExtractionLog.positiveFeedback)}
        </ReportSection>
    </div>
  );
};

export default AnalysisReport;