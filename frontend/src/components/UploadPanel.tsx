import { useState } from "react";

import type { DatasetSummary } from "../types/dataset";

interface UploadPanelProps {
  datasetSummary: DatasetSummary | null;
}

export function UploadPanel({ datasetSummary }: UploadPanelProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <aside className={`sidebar-panel ${mobileOpen ? "sidebar-open" : ""}`}>
      <button
        className="sidebar-toggle"
        onClick={() => setMobileOpen((prev) => !prev)}
        aria-expanded={mobileOpen}
      >
        <span>Dataset info</span>
        <span className={`sidebar-chevron ${mobileOpen ? "sidebar-chevron-open" : ""}`}>
          &#9662;
        </span>
      </button>

      <div className="sidebar-content">
        <div>
          <p className="eyebrow">Dataset</p>
          <h2>Rice Workbook</h2>
          <p className="sidebar-copy">
            Ask questions in natural language about the pre-loaded rice EXIM dataset.
          </p>
        </div>

        <div className="dataset-card">
          <strong>Current status</strong>
          {datasetSummary ? (
            <>
              <p>{datasetSummary.file_name}</p>
              <p>
                {datasetSummary.row_count} rows • {datasetSummary.column_count} columns
              </p>
              <p className="dataset-columns">{datasetSummary.columns.join(", ")}</p>
            </>
          ) : (
            <p>Loading dataset...</p>
          )}
        </div>
      </div>
    </aside>
  );
}
