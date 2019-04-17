// A* implementation sample written by Akihiko Yamaguchi

#include <list>
#include <vector>
#include <algorithm> // std::find
#include <cassert>

class TAStarSearch
{
public:
  typedef int TState;  // Type of node
  typedef double TValue;  // Type of cost
  static const TState InvalidState;

  TAStarSearch()
    : state_max_(InvalidState),
      start_(InvalidState),
      goal_(InvalidState),
      cost_function_(NULL),
      heuristic_cost_function_(NULL),
      successors_function_(NULL),
      callback_on_updated_(NULL)  {}

  // Solve the problem; return true if solved, false if failed
  bool Solve();

  /* Return a solved shortest path to goal.
      If goal is not speficied, the path to the stored goal is returned. */
  std::list<TState> GetSolvedPath(const TState &goal=InvalidState);

  // State space is {0,..,state_max_}
  void SetStateMax(const TState &state_max)  {state_max_= state_max;}
  void SetProblem(const TState &start, const TState &goal)  {start_= start; goal_= goal;}
  const TState& StateMax() const {return state_max_;}
  const TState& Start() const {return start_;}
  const TState& Goal() const {return goal_;}

  // Get a vector: each element(state) indicates a state where that state comes from
  const std::vector<TState>& InversePolicy() const {return inverse_policy_;}

  // User-defined cost funcion:
  void SetCostFunction(TValue (*f)(const TState &state1, const TState &state2))  {cost_function_= f;}
  // User-defined heuristic funcion:
  void SetHeuristicCostFunction(TValue (*f)(const TState &state1, const TState &state2))  {heuristic_cost_function_= f;}
  /* User-defined state-transition function that returns
      a set of possible successors from the current state */
  void SetSuccessorsFunction(std::list<TState> (*f)(const TState &state))  {successors_function_= f;}

  /* User-defined callback function executed when InversePolicy is updated.
      Edge state1 --> state2 shows a new one. */
  void SetCallbackOnUpdated(void (*f)(const TAStarSearch &a, const TState &state1, const TState &state2))  {callback_on_updated_= f;}

private:
  TState state_max_;

  TState start_, goal_;

  // Each element(state) indicates a state where that state comes from
  std::vector<TState>  inverse_policy_;

  // Function pointers
  TValue (*cost_function_)(const TState &state1, const TState &state2);
  TValue (*heuristic_cost_function_)(const TState &state1, const TState &state2);
  std::list<TState> (*successors_function_)(const TState &state);

  void (*callback_on_updated_)(const TAStarSearch &a, const TState &state1, const TState &state2);

  int get_state_num()  {return state_max_+1;}

  // Find a state ``s'' in ``set' where ``value[s]' is the lowest in ``value'
  TState find_lowest(const std::list<TState> &set, const std::vector<TValue> &value)
    {
      if(set.empty())  return InvalidState;
      std::list<TState>::const_iterator lowest= set.begin();
      for(std::list<TState>::const_iterator itr(lowest),last(set.end()); itr!=last; ++itr)
        if(value[*itr]<value[*lowest])  lowest= itr;
      return *lowest;
    }

  // Return true if ``item'' is included in ``set''
  bool is_in(const TState &item, const std::list<TState> &set)
    {
      return std::find(set.begin(),set.end(),item)!=set.end();
    }
};
const TAStarSearch::TState TAStarSearch::InvalidState(-1);


// Solve the problem; return true if solved, false if failed
bool TAStarSearch::Solve()
{
  // Check the configuration validity:
  assert(state_max_!=InvalidState);
  assert(start_!=InvalidState);
  assert(goal_!=InvalidState);
  assert(cost_function_);
  assert(heuristic_cost_function_);
  assert(successors_function_);

  std::list<TState> closedset; // The set of nodes already evaluated
  std::list<TState> openset;   // The set of tentative nodes to be evaluated,
  openset.push_back(start_);   // initially containing the start node

  // The map of navigated nodes:
  inverse_policy_.resize(get_state_num());
  std::fill(inverse_policy_.begin(), inverse_policy_.end(), InvalidState);

  std::vector<TValue>  g_score(get_state_num(), 0.0l);
  std::vector<TValue>  f_score(get_state_num(), 0.0l);
  g_score[start_]= 0.0l;   // Cost from start along best known path
  // Estimated total cost from start to goal
  f_score[start_]= g_score[start_] + heuristic_cost_function_(start_, goal_);


  /*
  *   ===========================================================
  *   IMPLEMENT THE REMAINING CODE BASED ON:
  *   http://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode
  *   ===========================================================
  */

  // Return false if the solution is not found
  return false;
}

/* Return a solved shortest path to goal.
    If goal is not speficied, the path to the stored goal is returned. */
std::list<TAStarSearch::TState> TAStarSearch::GetSolvedPath(const TState &goal)
{
  std::list<TState> path;
  TState current= ((goal!=InvalidState)?goal:goal_);
  path.push_front(current);
  while((current=inverse_policy_[current])!=InvalidState)
    path.push_front(current);
  return path;
}
