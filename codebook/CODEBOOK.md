# Construct codebook — Paper 13

Five TPB/UTAUT constructs. Every measurement arm (expert dictionary, topic
model, fine-tuned encoder, LLM) and every human coder is aimed at exactly
these definitions.

## Scoring

For each construct, record two numbers:

- **salience** (0–3): how much the document is *about* the construct.
  0 = not discussed at all; 3 = a central concern of the document.
- **level** (−3…+3): the level of the construct expressed.
  −3 = strongly low/negative, 0 = neutral or mixed, +3 = strongly high/positive.
  If salience is 0, leave level blank.

## ATT — Attitude toward the object

*Theory:* TPB (Ajzen, 1991); attitude toward the behaviour/object

**Definition.** The writer's overall evaluative judgement of the object (the venue, product, policy or actor) -- good/bad, pleasant/unpleasant, favourable/unfavourable -- irrespective of any specific benefit.

**Coding rule.** Code global evaluative language here. If the evaluation is tied to a specific functional outcome ('it removed the stain'), code PE instead; if it is tied to difficulty of use, code EE.

**Anchors.** −3 = Strongly unfavourable overall evaluation; +3 = Strongly favourable overall evaluation

**Expert lexicon (arm 1).**

- positive: `love`, `loved`, `great`, `excellent`, `amazing`, `awesome`, `wonderful`, `fantastic`, `outstanding`, `nice`, `good`, `lovely`, `delightful`, `enjoyed`, `enjoy`, `pleasant`, `pleased`, `happy with`, `satisfied`, `impressed`, `impressive`, `beautiful`, `charming`, `gem`, `favorite`, `favourite`, `perfect`, `best`, `solid`, `worth it`, `worth every`, `quality`, `top notch`, `superb`, `brilliant`, `fabulous`, `terrific`, `delicious`, `tasty`, `friendly`, `welcoming`, `cozy`, `clean`, `well made`, `well-made`
- negative: `terrible`, `awful`, `horrible`, `bad`, `poor`, `disappointing`, `disappointed`, `disappointment`, `mediocre`, `meh`, `unpleasant`, `nasty`, `gross`, `disgusting`, `rude`, `dirty`, `filthy`, `overpriced`, `rip off`, `ripoff`, `waste of money`, `waste of time`, `regret`, `hate`, `hated`, `dislike`, `unhappy`, `upset`, `angry`, `frustrating`, `frustrated`, `annoying`, `cheap feeling`, `flimsy`, `junk`, `garbage`, `trash`, `worst`, `avoid`, `sucks`, `sucked`, `not worth`, `unacceptable`, `appalling`, `atrocious`

## PE — Performance expectancy / perceived usefulness

*Theory:* UTAUT (Venkatesh et al., 2003); TAM perceived usefulness

**Definition.** The degree to which the writer believes the object performs its intended function and delivers the benefit it promises (it works, it is effective, it solves the problem).

**Coding rule.** Requires reference to a function, task or outcome. Pure affect ('I love it') is ATT, not PE.

**Anchors.** −3 = The object fails at its intended function; +3 = The object performs its intended function very well

**Expert lexicon (arm 1).**

- positive: `works`, `worked`, `works well`, `works great`, `works perfectly`, `does the job`, `did the job`, `does what`, `effective`, `efficient`, `useful`, `helpful`, `helps`, `helped`, `reliable`, `durable`, `long lasting`, `long-lasting`, `sturdy`, `powerful`, `accurate`, `consistent`, `performs`, `performance`, `solved`, `solves`, `fixed my`, `as advertised`, `as described`, `as expected`, `lives up to`, `gets the job done`, `no issues`, `no problems`, `results`, `improvement`, `improved`, `worth the`
- negative: `doesn't work`, `does not work`, `didn't work`, `did not work`, `stopped working`, `broke`, `broken`, `defective`, `faulty`, `useless`, `ineffective`, `unreliable`, `failed`, `fails`, `failure`, `malfunction`, `not as described`, `not as advertised`, `false advertising`, `fell apart`, `falls apart`, `leaks`, `leaked`, `error`, `errors`, `bug`, `bugs`, `glitch`, `crashes`, `crashed`, `no effect`, `did nothing`, `does nothing`, `waste`, `poor quality`, `low quality`

## EE — Effort expectancy / perceived ease of use

*Theory:* UTAUT effort expectancy; TPB perceived behavioural control

**Definition.** The degree of ease associated with obtaining, understanding, installing, operating or otherwise dealing with the object (clarity, simplicity, convenience, absence of hassle).

**Coding rule.** Waiting time, queueing, unclear instructions, complicated setup and bureaucratic burden all count as low EE.

**Anchors.** −3 = Dealing with the object is difficult, confusing or effortful; +3 = Dealing with the object is easy, clear and effortless

**Expert lexicon (arm 1).**

- positive: `easy`, `easy to use`, `easy to install`, `simple`, `straightforward`, `intuitive`, `user friendly`, `user-friendly`, `convenient`, `hassle free`, `hassle-free`, `effortless`, `quick`, `fast`, `smooth`, `seamless`, `clear instructions`, `well organized`, `well-organized`, `no hassle`, `painless`, `a breeze`, `plug and play`, `self explanatory`, `self-explanatory`, `prompt`, `efficient service`, `quick service`, `no wait`, `walk in`, `accessible`
- negative: `difficult`, `hard to`, `complicated`, `confusing`, `confused`, `complex`, `cumbersome`, `tedious`, `hassle`, `struggle`, `struggled`, `took forever`, `slow`, `waited`, `waiting`, `long wait`, `long line`, `long queue`, `unclear`, `not clear`, `counterintuitive`, `poorly written`, `instructions were`, `learning curve`, `inconvenient`, `impossible to`, `couldn't figure`, `could not figure`, `gave up`, `paperwork`, `bureaucratic`, `red tape`, `runaround`, `run around`

## SI — Social influence / subjective norm

*Theory:* TPB subjective norm; UTAUT social influence

**Definition.** Reference to what *other people* think, say, do or expect regarding the object: recommendations received, reputation, hype, reviews read, the behaviour of friends/family/colleagues, or normative pressure.

**Coding rule.** The referent must be a third party. The writer's own advice to the reader ('I recommend it') is BI (intention to recommend), not SI.

**Anchors.** −3 = Others discourage, warn against, or think badly of the object; +3 = Others encourage, praise or expect engagement with the object

**Expert lexicon (arm 1).**

- positive: `recommended by`, `recommended to me`, `friend recommended`, `my friend`, `my friends`, `my wife`, `my husband`, `my family`, `my colleague`, `coworker`, `everyone says`, `everyone said`, `everybody loves`, `people say`, `people said`, `word of mouth`, `heard about`, `heard great things`, `reviews said`, `the reviews`, `highly rated`, `popular`, `hype`, `well known`, `reputation`, `famous`, `raved`, `raving`, `told me about`, `was told`, `brought me here`, `came with`, `based on reviews`
- negative: `warned me`, `warned us`, `friend warned`, `bad reviews`, `negative reviews`, `people complain`, `everyone complains`, `don't believe the hype`, `overhyped`, `over hyped`, `overrated`, `not what people say`, `poor reputation`, `bad reputation`, `told me not to`, `advised against`

## BI — Behavioural intention

*Theory:* TPB / UTAUT behavioural intention

**Definition.** The writer's stated intention or commitment regarding future behaviour toward the object: returning, repurchasing, continuing, recommending to others, or refusing to do any of these.

**Coding rule.** Must be forward-looking or dispositional. A past visit is not BI; 'I'll be back' and 'I would recommend it' are.

**Anchors.** −3 = Explicit intention not to return / not to use / to warn others off; +3 = Explicit intention to return, repurchase, continue or recommend

**Expert lexicon (arm 1).**

- positive: `will be back`, `will come back`, `coming back`, `be back`, `will return`, `would return`, `go back`, `again and again`, `will buy again`, `would buy again`, `buy it again`, `purchase again`, `will order again`, `order again`, `reorder`, `repurchase`, `will definitely`, `definitely recommend`, `highly recommend`, `would recommend`, `i recommend`, `recommend it`, `recommend this`, `must try`, `must visit`, `check it out`, `give it a try`, `will keep`, `keep using`, `will continue`, `continue to use`, `new favorite`, `regular`, `loyal`, `subscribe`, `sign up`, `support this`, `in favor of`, `urge you to adopt`
- negative: `never again`, `won't be back`, `will not be back`, `not coming back`, `never coming back`, `won't return`, `will not return`, `won't buy again`, `would not buy`, `never buy`, `not recommend`, `would not recommend`, `do not recommend`, `don't recommend`, `wouldn't recommend`, `cannot recommend`, `steer clear`, `stay away`, `returning it`, `returned it`, `sent it back`, `refund`, `cancel`, `cancelled`, `canceled`, `unsubscribe`, `oppose`, `object to`, `urge you to reject`, `should be withdrawn`

## Directional disambiguation patterns

Applied on the raw string, after token matching:

- `\b(friend|friends|wife|husband|family|colleague|coworker|neighbou?r|sister|brother|mother|father|someone|somebody|they|he|she)\s+\w{0,12}\s?recommend` → **SI** (+1.0)
- `\b(was|were|got)\s+recommended\b` → **SI** (+1.0)
- `\bi\s+(would\s+|will\s+|do\s+|highly\s+|definitely\s+)*recommend` → **BI** (+1.0)

## Worked boundary cases

| Text | Correct coding |
|---|---|
| "My friend told me to try it" | SI (received), not BI |
| "I'd recommend it to anyone" | BI (given), not SI |
| "It smells lovely" | ATT, not PE |
| "It actually removed the stain" | PE, not ATT |
| "We waited 45 minutes" | EE (low), not ATT |
| "Returning it tomorrow" | BI (negative) |
