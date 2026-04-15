import { ChangeEvent, useState } from "react";

import type { DatasetSummary } from "../types/dataset";

interface UploadPanelProps {
  datasetSummary: DatasetSummary | null;
  uploading: boolean;
  onUpload: (file: File) => Promise<void>;
}

export function UploadPanel({ datasetSummary, uploading, onUpload }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFile(event.target.files?.[0] ?? null);
  }

  async function handleUploadClick() {
    if (!selectedFile) {
      return;
    }
    await onUpload(selectedFile);
    setSelectedFile(null);
  }

  return (
    <aside className="sidebar-panel">
      <div>
        <p className="eyebrow">Dataset</p>
        <h2>Rice Workbook</h2>
        <p className="sidebar-copy">
          Upload an Excel file to prepare the backend for grounded question answering.
        </p>
      </div>

      <label className="upload-field">
        <span>Select `.xlsx` file</span>
        <input type="file" accept=".xlsx,.xls" onChange={handleFileChange} disabled={uploading} />
      </label>

      <button className="upload-button" onClick={handleUploadClick} disabled={!selectedFile || uploading}>
        {uploading ? "Uploading..." : "Upload Dataset"}
      </button>

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
          <p>No dataset loaded yet.</p>
        )}
      </div>
    </aside>
  );
}
