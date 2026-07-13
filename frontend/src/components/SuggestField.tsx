"use client";

type SuggestFieldProps = {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  placeholder?: string;
  maxLength?: number;
};

// Text field with a native browser dropdown of common presentations, backed by
// <datalist> — staff can pick a suggestion or type any free text; both write to
// the same value, so nothing about the underlying field's meaning changes.
export default function SuggestField({
  id,
  label,
  value,
  onChange,
  options,
  placeholder,
  maxLength,
}: SuggestFieldProps) {
  const listId = `${id}-options`;
  return (
    <div style={{ gridColumn: "1 / -1" }}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        list={listId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        maxLength={maxLength}
        autoComplete="off"
      />
      <datalist id={listId}>
        {options.map((o) => (
          <option key={o} value={o} />
        ))}
      </datalist>
    </div>
  );
}
