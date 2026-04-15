import { apiGet, apiUpload } from "./api";
import type { DatasetLoadResponse, DatasetSummary } from "../types/dataset";

export function uploadDataset(file: File): Promise<DatasetLoadResponse> {
  return apiUpload<DatasetLoadResponse>("/datasets/upload", file);
}

export function fetchDatasetSummary(): Promise<DatasetSummary> {
  return apiGet<DatasetSummary>("/datasets/summary");
}
