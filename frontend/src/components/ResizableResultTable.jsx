import Table from "@arco-design/web-react/es/Table";
import { useLayoutEffect, useRef, useState } from "react";

const MIN_COLUMN_WIDTH = 96;

function ResizableHeaderCell({ children, resizeTitle, resizeWidth, resizeMinWidth, onResize, ...props }) {
  const drag = useRef(null);
  const [resizing, setResizing] = useState(false);

  function finishResize(event) {
    if (drag.current?.pointerId !== event.pointerId) return;
    drag.current = null;
    setResizing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return <th {...props}>
    {children}
    <span
      className="result-column-resizer"
      role="separator"
      tabIndex={0}
      aria-label={`调整${resizeTitle}列宽`}
      aria-orientation="vertical"
      aria-valuemin={resizeMinWidth}
      aria-valuenow={resizeWidth}
      aria-valuetext={`${resizeWidth} 像素`}
      title="拖动调整列宽；左右方向键调整，Shift 加快，Home 最小宽度"
      data-resizing={resizing || undefined}
      onPointerDown={(event) => {
        if (!event.isPrimary || event.button !== 0 || drag.current) return;
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { pointerId: event.pointerId, x: event.clientX, width: resizeWidth };
        setResizing(true);
      }}
      onPointerMove={(event) => {
        if (drag.current?.pointerId !== event.pointerId) return;
        onResize(Math.max(resizeMinWidth, Math.round(drag.current.width + event.clientX - drag.current.x)));
      }}
      onPointerUp={finishResize}
      onPointerCancel={finishResize}
      onLostPointerCapture={finishResize}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 64 : 16;
        const width = { ArrowLeft: resizeWidth - step, ArrowRight: resizeWidth + step, Home: resizeMinWidth }[event.key];
        if (width === undefined) return;
        event.preventDefault();
        event.stopPropagation();
        onResize(Math.max(resizeMinWidth, width));
      }}
    />
  </th>;
}

const TABLE_COMPONENTS = { header: { th: ResizableHeaderCell } };

export default function ResizableResultTable({ columns, widths, onWidthsChange, defaultColumnWidth = 180, ...props }) {
  const container = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useLayoutEffect(() => {
    const element = container.current;
    function measureContainer() {
      setContainerWidth(element.querySelector(".arco-table-content-inner").clientWidth);
    }
    measureContainer();
    const observer = new ResizeObserver(measureContainer);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    const missing = columns.filter((column) => widths[column.key] === undefined);
    if (!missing.length) return;
    const baseWidth = columns.reduce((total, column) => total + (widths[column.key] ?? column.width ?? defaultColumnWidth), 0);
    const flexible = missing.filter((column) => !column.width);
    const extraWidth = Math.max(0, container.current.clientWidth - baseWidth);
    const initialWidths = Object.fromEntries(missing.map((column) => [
      column.key,
      Math.max(MIN_COLUMN_WIDTH, Math.round((column.width ?? defaultColumnWidth) + (!column.width && flexible.length ? extraWidth / flexible.length : 0))),
    ]));
    onWidthsChange((current) => ({ ...initialWidths, ...current }));
  }, [columns, widths, onWidthsChange, defaultColumnWidth]);

  const requestedWidths = columns.map((column) => widths[column.key] ?? column.width ?? defaultColumnWidth);
  const lastColumnIndex = columns.length - 1;
  const precedingWidth = requestedWidths.slice(0, lastColumnIndex).reduce((total, width) => total + width, 0);
  const resizedColumns = columns.map((column, index) => {
    const minimumWidth = index === lastColumnIndex ? Math.max(MIN_COLUMN_WIDTH, containerWidth - precedingWidth) : MIN_COLUMN_WIDTH;
    const width = Math.max(requestedWidths[index], minimumWidth);
    return {
      ...column,
      width,
      onHeaderCell: () => ({
        resizeTitle: column.title,
        resizeWidth: width,
        resizeMinWidth: minimumWidth,
        onResize: (nextWidth) => onWidthsChange((current) => ({
          ...current,
          [column.key]: index === lastColumnIndex && nextWidth <= minimumWidth ? MIN_COLUMN_WIDTH : nextWidth,
        })),
      }),
    };
  });
  const tableWidth = resizedColumns.reduce((total, column) => total + column.width, 0);

  return <div ref={container} className="resizable-result-table">
    <Table {...props} columns={resizedColumns} components={TABLE_COMPONENTS} tableLayoutFixed scroll={{ x: tableWidth }} />
  </div>;
}
