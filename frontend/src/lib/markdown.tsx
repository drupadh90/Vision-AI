/**
 * Tiny dependency-free markdown renderer for agent output.
 *
 * Deliberately not `dangerouslySetInnerHTML`: agent output is untrusted text,
 * so everything is emitted as React elements and can never inject HTML.
 */
import { Fragment, type ReactNode } from "react";

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern =
    /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyPrefix}-i${i++}`;

    if (token.startsWith("`")) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("[")) {
      const linkMatch = /\[([^\]]+)\]\(([^)]+)\)/.exec(token);
      if (linkMatch) {
        const href = linkMatch[2];
        const safe = /^https?:\/\//i.test(href) ? href : "#";
        nodes.push(
          <a key={key} href={safe} target="_blank" rel="noreferrer noopener">
            {linkMatch[1]}
          </a>,
        );
      }
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function Markdown({ children }: { children: string }) {
  const lines = (children ?? "").split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let ordered = false;
  // Explicitly typed: TS narrows `code` to `never` inside the loop otherwise.
  let code: string[] | null = null as string[] | null;
  let codeLang = "";

  const flushList = (key: string) => {
    if (!list.length) return;
    const items = list.map((item, i) => <li key={i}>{renderInline(item, `${key}-${i}`)}</li>);
    blocks.push(ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>);
    list = [];
  };

  lines.forEach((raw, idx) => {
    const line = raw.replace(/\s+$/, "");
    const key = `b${idx}`;

    if (line.trim().startsWith("```")) {
      if (code === null) {
        flushList(`${key}-l`);
        code = [];
        codeLang = line.trim().slice(3);
      } else {
        blocks.push(
          <pre key={key}>
            <code data-lang={codeLang}>{code.join("\n")}</code>
          </pre>,
        );
        code = null;
        codeLang = "";
      }
      return;
    }
    if (code !== null) {
      code.push(raw);
      return;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushList(`${key}-l`);
      const level = heading[1].length;
      const content = renderInline(heading[2], key);
      blocks.push(
        level === 1 ? <h1 key={key}>{content}</h1> :
        level === 2 ? <h2 key={key}>{content}</h2> :
        <h3 key={key}>{content}</h3>,
      );
      return;
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet) {
      if (ordered) flushList(`${key}-l`);
      ordered = false;
      list.push(bullet[1]);
      return;
    }
    if (numbered) {
      if (!ordered) flushList(`${key}-l`);
      ordered = true;
      list.push(numbered[1]);
      return;
    }

    flushList(`${key}-l`);
    if (line.trim() === "") return;
    blocks.push(<p key={key}>{renderInline(line, key)}</p>);
  });

  flushList("tail");
  if (code !== null) {
    blocks.push(
      <pre key="tail-code">
        <code>{code.join("\n")}</code>
      </pre>,
    );
  }

  return <div className="md">{blocks.map((b, i) => <Fragment key={i}>{b}</Fragment>)}</div>;
}
