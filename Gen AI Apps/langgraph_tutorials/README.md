# LangGraph Tutorials

1. Basics
2. Tool Calling
3. Agents Basics
4. RAG Basics
5. RAG Agents


When to use LangGraph

| Use LangChain When | Use LangGraph When |
| ------------------ | ------------------ |
| Simple Chains | Stateful Workflows |
| One shot QnA | Mult-Turn Agents |
| RAG Pipelines | Self correcting loops |
| No loops needed | Human in the loop |
| Prototyping Phase | Production Phase |


## Real World Use cases

- Customer Support Bot: Handles Tickets, escalates, and follows up
- Research Agent: Searches, Evaluates, Iterates until satisfied
- Code Review: Analyses, Suggests, re-checks after changes
- Approval Workflows: Multi-day processes with human review

## The Three Pillars of LangGraph

- State: What the agent knows/tracks
- Nodes: Functions that process state
- Edges: Connections between nodes

## StateGraph

- Define State
    - AgentState(TypedDict)
- Create Graph
    - StateGraph(AgentState)
- Add Nodes
    - graph.add_node("process", fn)
- Add Edges
    - graph.add_edge(START, ...)
- Compile
    - app = graph.compile()

### State
- Simple field
    - current_step: str -> Replaced each time: old => new
- Annotated(add_messages)
    - messages: Annotated[list, add_messages] -> Values accumulated: [a] + [b] => [a, b]
- Annotated(operatora.add)
    - token_count: Annotated[int, operator.add] -> Numbers sum: 5+3 => 8

### Nodes
- Nodes receive full state, return partial updates
- Input: Full State
    - state['messages']
- Process
    - Read state -> Do work (LLM Call, etc..) -> return updates
- Output
    - return {"messages": [resp]}

### Edges
- Direct Edge
    - A always goes to B
        - graph.add_edge("A", "B")
- Conditional Edge
    - A goes to B or C based on logic
        - graph.add_conditional_edges("A", routing_fn, {"route_b": "B", "route_c": "C"})
