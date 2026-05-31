/**
 * History Page
 * ============
 * Paginated list of past predictions with search and filtering.
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiService } from "@/lib/api";
import type { PredictionRecord, HistoryParams } from "@/lib/types";
import { ChevronLeft, ChevronRight, Search, Eye, Trash2 } from "lucide-react";

export default function HistoryPage() {
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  const pageSize = 20;

  const fetchHistory = async (params: HistoryParams) => {
    setLoading(true);
    try {
      const result = await apiService.getHistory(params);
      setPredictions(result.items);
      setTotal(result.total);
      setTotalPages(result.total_pages);
    } catch (error) {
      console.error("Failed to load history:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory({ page, page_size: pageSize, search: search || undefined });
  }, [page, search]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchHistory({ page: 1, page_size: pageSize, search: search || undefined });
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this prediction?")) return;
    try {
      await apiService.deletePrediction(id);
      fetchHistory({ page, page_size: pageSize, search: search || undefined });
    } catch (error) {
      console.error("Failed to delete prediction:", error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">History</h1>
          <p className="text-gray-500 mt-1">{total} predictions</p>
        </div>
      </div>

      {/* Search */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by filename or OCR text..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-400"
          />
        </div>
        <button
          type="submit"
          className="px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 transition-colors"
        >
          Search
        </button>
      </form>

      {/* Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
          </div>
        ) : predictions.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-500">
            No predictions found.{" "}
            <Link href="/predict" className="text-primary-400 hover:underline">
              Make your first prediction
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Filename</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Theme</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Sentiment</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {predictions.map((pred) => (
                  <tr key={pred.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 text-sm text-gray-900 font-medium">
                      {pred.filename || "Untitled"}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">{pred.theme || "—"}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{pred.sentiment || "—"}</td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {pred.processing_ms ? `${pred.processing_ms}ms` : "—"}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(pred.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          href={`/history/${pred.id}`}
                          className="p-1 hover:bg-gray-100 rounded transition-colors"
                          title="View details"
                        >
                          <Eye className="w-4 h-4 text-gray-400" />
                        </Link>
                        <button
                          onClick={() => handleDelete(pred.id)}
                          className="p-1 hover:bg-red-50 rounded transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-500" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500">
            Page {page} of {totalPages}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="p-2 hover:bg-gray-100 rounded-lg disabled:opacity-50 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="p-2 hover:bg-gray-100 rounded-lg disabled:opacity-50 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
