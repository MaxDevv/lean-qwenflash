"""Separate streamable prose from a possibly fragmented native tool call."""

def split_tool_stream(pending):
    """Return (safe prose, held suffix); never expose native tool-call markup.

    Hold only a possible opening-tag prefix until a tool call actually starts.
    Once it starts, retain the remaining text for the existing full-call parser.
    """
    marker='<tool_call>'
    index=pending.find(marker)
    if index>=0:
        return pending[:index],pending[index:]
    for length in range(min(len(marker)-1,len(pending)),0,-1):
        if marker.startswith(pending[-length:]):
            return pending[:-length],pending[-length:]
    return pending,''
