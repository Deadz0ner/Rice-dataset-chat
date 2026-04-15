export interface DatasetSummary {
  file_name: string;
  row_count: number;
  column_count: number;
  columns: string[];
  sample_rows: Record<string, string>[];
}

export interface DatasetLoadResponse {
  message: string;
  summary: DatasetSummary;
}
