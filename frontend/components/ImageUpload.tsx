'use client';

/**
 * Image Upload Component
 * ======================
 * Drag & drop + click to upload interface for images.
 * Supports multiple files with preview.
 */

import { useCallback, useState, useRef } from 'react';
import { Upload, X, } from 'lucide-react';
import { UploadFile } from '@/lib/types';
import Image from 'next/image';
interface ImageUploadProps {
  onFilesSelected: (files: UploadFile[]) => void;
  disabled?: boolean;
  maxFiles?: number;
}

export default function ImageUpload({
  onFilesSelected,
  disabled = false,
  maxFiles = 10,
}: ImageUploadProps) {
  // State
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Ref for file input
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Validate and process selected files.
   */
  const processFiles = useCallback((fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) {
      return;
    }

    setError(null);

    // Convert FileList to array
    const newFiles = Array.from(fileList);

    // Check total count
    const totalFiles = files.length + newFiles.length;
    if (totalFiles > maxFiles) {
      setError(`Maximum ${maxFiles} files allowed. You selected ${totalFiles} files.`);
      return;
    }

    // Validate file types
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp'];
    const invalidFiles = newFiles.filter(file => !allowedTypes.includes(file.type));
    
    if (invalidFiles.length > 0) {
      setError(`Invalid file type(s): ${invalidFiles.map(f => f.name).join(', ')}`);
      return;
    }

    // Check file sizes (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB
    const oversizedFiles = newFiles.filter(file => file.size > maxSize);
    
    if (oversizedFiles.length > 0) {
      setError(`File(s) too large: ${oversizedFiles.map(f => f.name).join(', ')}. Max 10MB per file.`);
      return;
    }

    // Create preview URLs and UploadFile objects
    const uploadFiles: UploadFile[] = newFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      id: `${file.name}-${Date.now()}-${Math.random()}`,
    }));

    // Update state
    const updatedFiles = [...files, ...uploadFiles];
    setFiles(updatedFiles);
    onFilesSelected(updatedFiles);

  }, [files, maxFiles, onFilesSelected]);

  /**
   * Handle file input change.
   */
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(e.target.files);
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  /**
   * Handle drag over event.
   */
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  /**
   * Handle drag leave event.
   */
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  /**
   * Handle drop event.
   */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (!disabled) {
      processFiles(e.dataTransfer.files);
    }
  };

  /**
   * Handle click to open file dialog.
   */
  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  /**
   * Remove a file from the list.
   */
  const handleRemoveFile = (id: string) => {
    const updatedFiles = files.filter(f => f.id !== id);
    setFiles(updatedFiles);
    onFilesSelected(updatedFiles);
    
    // Revoke preview URL to free memory
    const file = files.find(f => f.id === id);
    if (file) {
      URL.revokeObjectURL(file.preview);
    }
  };

  /**
   * Clear all files.
   */
  const handleClearAll = () => {
    // Revoke all preview URLs
    files.forEach(file => URL.revokeObjectURL(file.preview));
    
    setFiles([]);
    onFilesSelected([]);
    setError(null);
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-all duration-200
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-500 hover:bg-blue-50'}
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white'}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/jpg,image/png,image/bmp,image/webp"
          onChange={handleFileInputChange}
          disabled={disabled}
          className="hidden"
        />

        <Upload className={`mx-auto h-12 w-12 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
        
        <h3 className="mt-4 text-lg font-medium text-gray-900">
          {isDragging ? 'Drop files here' : 'Upload Images'}
        </h3>
        
        <p className="mt-2 text-sm text-gray-600">
          Drag and drop images here, or click to browse
        </p>
        
        <p className="mt-1 text-xs text-gray-500">
          JPG, PNG, BMP, WEBP up to 10MB • Maximum {maxFiles} files
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* File Previews */}
      {files.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-gray-900">
              Selected Images ({files.length})
            </h4>
            <button
              onClick={handleClearAll}
              disabled={disabled}
              className="text-sm text-red-600 hover:text-red-700 disabled:opacity-50"
            >
              Clear All
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {files.map((file) => (
              <div
                key={file.id}
                className="relative group rounded-lg overflow-hidden border border-gray-200 bg-white"
              >
                {/* Image Preview */}
                <div className="aspect-square bg-gray-100">
                  <Image
                    src={file.preview}
                    alt={file.file.name}
                    fill unoptimized
                    className="w-full h-full object-cover"
                  />
                </div>
                

                {/* File Info */}
                <div className="p-2">
                  <p className="text-xs text-gray-900 truncate" title={file.file.name}>
                    {file.file.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(file.file.size / 1024).toFixed(1)} KB
                  </p>
                </div>

                {/* Remove Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveFile(file.id);
                  }}
                  disabled={disabled}
                  className="z-10 absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 disabled:opacity-50"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}