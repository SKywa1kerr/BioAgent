import type { CSSProperties } from "react";
import type { AlignmentRange, AlignmentViewModel, MutationRange } from "./types";
import "./SequenceAlignmentView.css";

interface Props {
  view: AlignmentViewModel;
}

function rangeStyle(range: AlignmentRange): CSSProperties {
  return {
    "--range-start": range.start,
    "--range-size": Math.max(1, range.end - range.start),
  } as CSSProperties;
}

function mutationTitle(mutation: MutationRange): string {
  return mutation.effect ? `${mutation.label} (${mutation.effect})` : mutation.label;
}

function markerTone(mutation: MutationRange): string {
  return (mutation.effect || mutation.type || "mutation").replace(/[^a-z0-9_-]/gi, "-");
}

export function SequenceAlignmentView({ view }: Props) {
  const alignmentStyle = {
    "--alignment-length": view.refLine.length,
  } as CSSProperties;

  return (
    <div className="sequence-alignment-view">
      {view.aaChanges.length > 0 ? (
        <div className="sequence-alignment-view__aa" aria-label="AA changes">
          {view.aaChanges.map((change) => (
            <span className="sequence-alignment-view__aa-pill" key={change}>
              {change}
            </span>
          ))}
        </div>
      ) : null}

      <div className="sequence-alignment-view__scroller">
        <div className="sequence-alignment-view__alignment" style={alignmentStyle}>
          <div className="sequence-alignment-view__track" aria-hidden="true">
            {view.cdsRange ? (
              <div className="sequence-alignment-view__cds" style={rangeStyle(view.cdsRange)} title="CDS" />
            ) : null}
            {view.mutationRanges.map((mutation, index) => (
              <div
                className={`sequence-alignment-view__marker sequence-alignment-view__marker--${markerTone(mutation)}`}
                key={`${mutation.start}-${mutation.end}-${mutation.label}-${index}`}
                style={rangeStyle(mutation)}
                title={mutationTitle(mutation)}
              />
            ))}
          </div>

          <div className="sequence-alignment-view__row">
            <span className="sequence-alignment-view__label">REF</span>
            <span className="sequence-alignment-view__line">{view.refLine}</span>
          </div>
          <div className="sequence-alignment-view__row">
            <span className="sequence-alignment-view__label">MATCH</span>
            <span className="sequence-alignment-view__line">{view.matchLine}</span>
          </div>
          <div className="sequence-alignment-view__row">
            <span className="sequence-alignment-view__label">QRY</span>
            <span className="sequence-alignment-view__line">{view.queryLine}</span>
          </div>
          <div className="sequence-alignment-view__row">
            <span className="sequence-alignment-view__label">POS</span>
            <span className="sequence-alignment-view__line">{view.positionLine}</span>
          </div>
          <div className="sequence-alignment-view__row">
            <span className="sequence-alignment-view__label" aria-hidden="true" />
            <span className="sequence-alignment-view__line">{view.tickLine}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
