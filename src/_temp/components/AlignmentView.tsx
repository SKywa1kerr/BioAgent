import React from "react";
import { AlignmentResult } from "../types";
import "./AlignmentView.css";

interface AlignmentViewProps {
  result: AlignmentResult | null;
  startPosition?: number;
  endPosition?: number;
}

export const AlignmentView: React.FC<AlignmentViewProps> = ({
  result,
  startPosition = 1,
  endPosition,
}) => {
  if (!result) {
    return <div className="alignment-view">No alignment data</div>;
  }

  const refSeq = result.refSequence;
  const querySeq = result.alignedQuery;
  const matches = result.matches;

  const end = endPosition || refSeq.length;
  const displayLength = Math.min(60, end - startPosition + 1);

  const renderSequence = (seq: string, label: string) => {
    const chars = seq.slice(startPosition - 1, startPosition - 1 + displayLength);
    return (
      <div className="sequence-row">
        <span className="sequence-label">{label}</span>
        <span className="sequence-chars">
          {chars.split("").map((char, i) => (
            <span key={i} className="base">{char}</span>
          ))}
        </span>
      </div>
    );
  };

  const renderMatch = () => {
    const matchChars = matches.slice(startPosition - 1, startPosition - 1 + displayLength);
    return (
      <div className="sequence-row match-row">
        <span className="sequence-label"> </span>
        <span className="sequence-chars">
          {matchChars.map((match, i) => (
            <span key={i} className={`match-char ${match ? "match" : "mismatch"}`}>
              {match ? "|" : "✕"}
            </span>
          ))}
        </span>
      </div>
    );
  };

  return (
    <div className="alignment-view">
      <div className="position-row">
        <span className="sequence-label"> </span>
        <span className="position-numbers">
          {Array.from({ length: displayLength }, (_, i) => {
            const pos = startPosition + i;
            return pos % 10 === 0 ? (
              <span key={i} className="position-mark">{pos}</span>
            ) : (
              <span key={i} className="position-spacer"> </span>
            );
          })}
        </span>
      </div>
      {renderSequence(querySeq, "Query")}
      {renderMatch()}
      {renderSequence(refSeq, "Ref  ")}
    </div>
  );
};
