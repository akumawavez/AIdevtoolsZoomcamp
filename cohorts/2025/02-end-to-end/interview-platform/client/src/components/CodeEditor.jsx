import React, { useState } from 'react';
import Editor from '@monaco-editor/react';

const CodeEditor = ({ code, setCode, language = "javascript" }) => {
  const handleEditorChange = (value, event) => {
    setCode(value);
  };

  return (
    <div className="editor-container" style={{ height: "100%", width: "100%" }}>
      <Editor
        height="100%"
        defaultLanguage={language}
        defaultValue="// Start coding here"
        value={code}
        onChange={handleEditorChange}
        theme="vs-dark"
        options={{
            minimap: { enabled: false },
            fontSize: 14,
        }}
      />
    </div>
  );
};

export default CodeEditor;
