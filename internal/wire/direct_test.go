// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"testing"

	"github.com/buger/jsonparser"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateObject(t *testing.T) {
	cases := map[string][]byte{
		"empty":           nil,
		"null":            []byte("null"),
		"array":           []byte("[]"),
		"trailing":        []byte(`{"a":1} {}`),
		"duplicate":       []byte(`{"a":1,"a":2}`),
		"malformedNested": []byte(`{"a":{"b":[}`),
		"invalidUTF8":     []byte{'{', '"', 'a', '"', ':', '"', 0xff, '"', '}'},
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateObject(body)
			require.Error(t, err)
			if name == "null" || name == "array" {
				assert.ErrorIs(t, err, ErrNotObject)
			}
		})
	}

	require.NoError(t, ValidateObject([]byte(`{"a":[1,{"b":"c"}]}`)))
}

func TestDirectScalarValues(t *testing.T) {
	value, typ := directTestValue(t, `"line\n\u2603"`)
	text, err := String(value, typ)
	require.NoError(t, err)
	assert.Equal(t, "line\n☃", text)

	value, typ = directTestValue(t, `null`)
	text, err = String(value, typ)
	require.NoError(t, err)
	assert.Empty(t, text)

	value, typ = directTestValue(t, `true`)
	flag, err := Bool(value, typ)
	require.NoError(t, err)
	assert.True(t, flag)

	value, typ = directTestValue(t, `3`)
	count, err := Int(value, typ)
	require.NoError(t, err)
	assert.Equal(t, 3, count)

	value, typ = directTestValue(t, `1.0`)
	_, err = Int(value, typ)
	assert.Error(t, err)

	value, typ = directTestValue(t, `1.5`)
	rate, err := Float(value, typ)
	require.NoError(t, err)
	assert.Equal(t, 1.5, rate)

	value, typ = directTestValue(t, `["a",null,"☃"]`)
	values, err := Strings(value, typ)
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "", "☃"}, values)

	value, typ = directTestValue(t, `"nope"`)
	_, err = Bool(value, typ)
	assert.Error(t, err)
}

func TestFileMediaValue(t *testing.T) {
	value, typ := directTestValue(t, `{"file_id":"file-1","filename":"a.txt"}`)
	media, filename, err := ParseFileMediaValue(Value{Raw: value, Type: typ}, "file")
	require.NoError(t, err)
	assert.Equal(t, "file-1", media.Ref)
	assert.Equal(t, "a.txt", filename)
}

func directTestValue(t *testing.T, raw string) ([]byte, jsonparser.ValueType) {
	t.Helper()
	value, typ, _, err := jsonparser.Get([]byte(raw))
	require.NoError(t, err)
	return value, typ
}
