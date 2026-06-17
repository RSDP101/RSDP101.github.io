# Welcome to the blog

*June 16, 2026*

I've been meaning to start writing for a while. The plan is simple: short essays about the things I'm thinking about, mostly at the intersection of large language models and mathematics — sometimes leaning toward research, sometimes toward markets, sometimes toward whatever I happened to break that week.

## Why bother

There are already a lot of blogs about LLMs. There are even more about math. Why add another one?

A few reasons:

- **Writing is a forcing function.** If I can't explain why a method works in plain English, I probably don't understand it. The fastest way to find the holes in my own reasoning is to try to write it down and notice where the prose gets vague.
- **I work in an underwritten niche.** "LLMs that do real math" sits between two communities that don't always talk to each other — the math olympiad / formal proofs world and the ML scaling world. The blog is partly an attempt to bridge that.
- **Selfish reason.** When I learn something useful from a paper, I forget half of it within a month. Notes help.

## What to expect

Posts will land somewhere on the spectrum between:

1. **Quick takes** — a half-page on a single paper I found interesting, or a single idea that surprised me.
2. **Deeper dives** — longer pieces where I work through a method, usually with code or a small experiment. These will go up less often.
3. **Field reports** — things I've actually built, what worked, what didn't, what I'd do differently.

I'm not aiming for any particular cadence. One post a week when I have something to say; nothing for a month when I don't.

## A small example

Just to show that the formatting works, here's a code block:

```python
def is_pareto_dominant(point, frontier):
    """Return True if no point in `frontier` dominates `point`."""
    cost, perf = point
    for c, p in frontier:
        if c <= cost and p >= perf and (c, p) != (cost, perf):
            return False
    return True
```

And a blockquote, because every blog post needs one eventually:

> The point of writing is not to have written. It is to find out what you think.

## What's next

The next post will probably be about something concrete — maybe a walk-through of the Pareto frontier figure from *Escaping the Cognitive Well*, or a note on why blueprint refinement helps in formal theorem proving. Whichever I get bored with first.

If you have something you'd like me to write about, find me at [rodrigaosdp@gmail.com](mailto:rodrigaosdp@gmail.com).

— Rodrigo
