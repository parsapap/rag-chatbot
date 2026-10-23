export type ServerEvent = { event: string; data: unknown };

// Splits complete "event: x\ndata: y\n\n" blocks off the buffer; the unfinished tail waits for more bytes.
export function parseEvents(buffer: string): { events: ServerEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.filter(Boolean).map((block) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { event, data: JSON.parse(data) as unknown };
  });
  return { events, rest };
}

// The model sometimes writes citations with Persian or Arabic-Indic digits, like [۱].
const CITATION = /\[([0-9۰-۹٠-٩]+)\]/g;

const toNumber = (digits: string) =>
  Number(
    digits
      .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x6f0))
      .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x660)),
  );

// Text pieces stay strings; each [n] becomes the number n.
export function splitCitations(text: string): (string | number)[] {
  return text
    .split(CITATION)
    .map((part, i) => (i % 2 ? toNumber(part) : part))
    .filter((part) => part !== "");
}
